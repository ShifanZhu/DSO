/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0,0), thisToNext(SE3())
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww*hh];
	JbBuffer_new = new Vec10f[ww*hh];


	frameID=-1;
	fixAffine=true;
	printDebug=false;

	//! 这是
	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}


// CoarseInitializer::trackFrame 中将所有 points （第一帧上的点）的逆深度初始化为1。从金字塔最高层到最底层依次匹配，每一层的匹配都是高斯牛顿优化过程，
// 在 CoarseIntializer::calcResAndGS 中计算Hessian矩阵等信息，计算出来的结果在 CoarseInitializer::trackFrame 中更新相对位姿（存储在局部变量中，
// 现在还没有确定要不要接受这一步优化），在 CoarseInitializer::trackFrame 中调用 CoarseInitializer::doStep 中更新点的逆深度信息。
// 随后再调用一次 CoarseIntializer::calcResAndGS ，计算新的能量，如果新能量更低，那么就接受这一步优化，
// 在 CoarseInitializer::applyStep 中生效前面保存的优化结果。

// 一些加速优化过程的操作：1.每一层匹配开始的时候，调用一次 CoarseInitializer::propagateDown ，将当前层所有点的逆深度设置为的它们parent（上一层）的逆深度；
// 2. 在每次接受优化结果，更新每个点的逆深度，调用一次 CoarseInitializer::optReg 将所有点的 iR 设置为其 neighbour 逆深度的中位数，其实这个函数
// 在 CoarseInitializer::propagateDown 和 CoarseInitializer::propagateUp 中都有调用，iR 变量相当于是逆深度的真值，在优化的过程中，使用这个值计算
// 逆深度误差，效果是幅面中的逆深度平滑。

// 优化过程中的lambda和点的逆深度有关系，起一个加权的作用，也不是很明白对lambda增减的操作。在完成所有层的优化之后，进行 CoarseInitializer::propagateUp操作，
// 使用低一层点的逆深度更新其高一层点parent的逆深度，这个更新是基于iR的，使得逆深度平滑。高层的点逆深度，在后续的操作中，没有使用到，所以这一步操作我认为是无用的。

// 将当前帧赋给newFrame为第一次跟踪，并为第一次跟踪做准备，（此时snapped为false，上一篇博客有提到）将两帧相对位姿的平移部分置0，并且给第一帧选取的像素点相关信息设置初值。
// 该函数只有在程序最开始调用
bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	//! 只有在程序最开始的时候执行
	std::cout << "track frame!!!" << std::endl;
	newFrame = newFrameHessian;
//[ ***step 1*** ] 先显示新来的帧
	// 新的一帧, 在跟踪之前显示的
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushLiveFrame(newFrameHessian);

	int maxIterations[] = {5,5,10,30,50};


	//? 调参
	// 这个是位移的阈值，如果平移的总偏移量超过2.5 / 150 就认为此帧是snapped为true的帧了.
	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8;//*freeDebugParam4; // 近邻点对当前点逆深度的影响权重
	couplingWeight = 1;//*freeDebugParam5;

//[ ***step 2*** ] 初始化每个点逆深度为1, 初始化光度参数, 位姿SE3
	// 将当前帧赋给newFrame为第一次跟踪，并为第一次跟踪做准备，（此时snapped为false，上一篇博客有提到）将两帧相对位姿的平移部分置0，
	// 并且给第一帧选取的像素点相关信息设置初值。
	//对points点中的几个数据初始化过程
	//只要出现过足够大的位移后 就不再对其初始化，直接拿着里面的值去连续优化5次
	if(!snapped) //! snapped应该指的是位移足够大了，不够大就重新优化
	{
		// 初始化
		thisToNext.translation().setZero();
		for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		{
			// points 存储每一层上的点类, 是第一帧提取出来的，存储的是满足灰度梯度阈值的点
			// 在对每层图像的灰度梯度选点之后，将点存储到 points[lvl]，numPoints[lvl]表示lvl层选取的符合像素梯度阈值的像素点数量。
			int npts = numPoints[lvl];
			Pnt* ptsl = points[lvl];
			for(int i=0;i<npts;i++)
			{
				// 初始化每一个特征点的逆深度的期望值,该点在新的一帧(当前帧)上的逆深度,逆深度的Hessian, 即协方差
				ptsl[i].iR = 1; // 每一个特征点的逆深度的期望值初始化为1
				ptsl[i].idepth_new = 1; // 该点对应参考帧的逆深度的新值，_new的含义是指刚计算的新值，如果确定该新值更好之后才会apply到idepth
				ptsl[i].lastHessian = 0; // 逆深度的Hessian, 即协方差, dd*dd
			}
		}
	}


  // 设置两帧之间的相对位姿变换以及由曝光时间设置两帧的光度放射变换。
	SE3 refToNew_current = thisToNext; // 参考帧与当前帧之间位姿
	AffLight refToNew_aff_current = thisToNext_aff; // 参考帧与当前帧之间光度系数

	// 如果都有仿射系数, 则估计一个初值
	//firstFrame 是第一帧图像数据信息
	//如果无光度标定那么曝光时间ab_exposure就是1.那么下面这个就是 a= 0 b = 0
	if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0)
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // coarse approximation.


	// 金字塔跟踪模型（重点）：对金字塔每层进行跟踪，由最高层开始，构建残差进行优化。
	Vec3f latestRes = Vec3f::Zero();
	// 从顶层开始估计
	for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--)
	{

//[ ***step 3*** ] 使用计算过的上一层来初始化下一层
		// 顶层未初始化到, reset来完成
		// 对其他层：首先propagateDown(lvl+1);，对当前层的坏点继承其parent的逆深度信息。其他与最高层一样，也是先计算 resOld ，然后求解迭代增量，
		// 计算resNew,，如果energy减小则接受更新，继续迭代。如果增大则调整lambad，重新计算。
		if(lvl<pyrLevelsUsed-1)
			propagateDown(lvl+1); // 注意此处传入的为lvl+1

		// 最高层：
		// 首先利用 resetPoints(lvl);设置最高层选取的点的energy和idepth_new；pts[i].energy.setZero();pts[i].idepth_new = pts[i].idepth;
		// 并且将坏点的逆深度为当前点的neighbours的逆深度平均值。
		// 然后利用resOld = calcResAndGS()计算当前的残差energy和优化的H矩阵和b以及Hsc，bsc。其中的"sc"是指 Schur Complement
		// applyStep(lvl);应用计算的相关信息。
		Mat88f H,Hsc; Vec8f b,bsc;
		resetPoints(lvl); // 这里对顶层进行初始化!如果不是顶层则只把pts[i].energy置零和pts[i].idepth_new置为idepth
//[ ***step 4*** ] 迭代之前计算能量, Hessian等
		// calcResAndGS()是该函数优化部分的核心
		// H b: 对应"Gauss Newton 方程可以进一步写成"下边的公式，此处应该只更新了大 H 矩阵右下角 Jx21*Jx21 ，和　b 的下边　Jx21 * r21
		// Hsc bsc: 对应"Schur Complement 消除"下边的公式，此处应该只更新了大 H 矩阵右下角 的被减数部分，和　b 的下边的被减数部分
		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
		applyStep(lvl); // 新的能量付给旧的

		float lambda = 0.1;
		float eps = 1e-4;
		int fails=0;
		// 初始信息
		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					lvl, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])), // 卡方(res*res)平均值
					sqrtf((float)(resOld[1] / resOld[2])), // 逆深度能量平均值
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
		}

//[ ***step 5*** ] 迭代求解
		int iteration=0;
		while(true)
		{
//[ ***step 5.1*** ] 计算边缘化后的Hessian矩阵, 以及一些骚操作
			Mat88f Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda); // 这不是LM么,论文说没用, 嘴硬
			// 舒尔补, 边缘化掉逆深度状态
			// Hl 对应"Schur Complement 消除"下边的H矩阵右下角的式子
			Hl -= Hsc*(1/(1+lambda)); // 因为dd必定是对角线上的, 所以也乘倒数
			// bl 对应"Schur Complement 消除"下边的b矩阵下角的式子
			Vec8f bl = b - bsc*(1/(1+lambda));
			//? wM 为什么这么乘, 它对应着状态的 SCALE
			//? (0.01f/(w[lvl]*h[lvl]))是为了减小数值, 更稳定?
			Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
			bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));

//[ ***step 5.2*** ] 求解增量
			Vec8f inc;
			if(fixAffine) // 固定光度参数，只用前６个，最后两个置零
			{
				// wM 可能用来在对角存放，逆深度、旋转量、平移量、相机焦距、光度仿射系数等的比例系数
				inc.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (Hl.topLeftCorner<6,6>().ldlt().solve(bl.head<6>())));
				inc.tail<2>().setZero();
			}
			// 通过上述计算的矩阵信息，进行迭代增量的求解。 inc 迭代增量表示相对位姿增量，相对仿射变换增量（8维）。求解增量会加入lambda，有点类似与LM的求解方法。
			else
			{	// 不固定光度系数，８个全部使用
				inc = - (wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b. 迭代增量的求解
			}

//[ ***step 5.3*** ] 更新状态, doStep中更新逆深度
			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doStep(lvl, lambda, inc); // 不判断是否OK就直接用了吗？此处应该只是do一下，如果残差变小才accept，才最终apply

			// 利用doStep(lvl, lambda, inc);计算选取的像素点逆深度增量。
			// 再计算并更新了所有增量之后，重新计算残差以及优化用到的矩阵信息。其中calcEC(lvl)计算像素点的逆深度energy。
			// regEnergy[0]存的是老点的(逆深度-逆深度均值)的平方， regEnergy[1]存的是新点的(逆深度-逆深度均值)的平方， regEnergy[2]存储累加的次数

//[ ***step 5.4*** ] 计算更新后的能量并且与旧的对比判断是否accept
			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			// resNew[0] 存储的能量值(也就是pattern点的残差和，残差由当前帧和参考帧的灰度值相减得到)； resNew[1] 存储平移的能量值，相当于平移越大, 越容易初始化成功； resNew[2] 存储累加的点的个数
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			Vec3f regEnergy = calcEC(lvl);

			// 然后比较两次的energy，如果减小了则接受优化，并且将新计算的H矩阵和b矩阵赋值，从而实现迭代优化。
			// 反之则调整lambda的值继续，并且失败次数加1。
			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);


			bool accept = eTotalOld > eTotalNew;

			if(printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
			}
//[ ***step 5.5*** ] 接受的话, 更新状态,; 不接受则增大lambda
			if(accept)
			{
				//? 这是啥   答：应该是位移足够大，才开始优化IR
				// alphaK 的值为 2.5*2.5, numPoints[lvl]表示lvl层选取的像素点数量
				if(resNew[1] == alphaK*numPoints[lvl]) // 当 alphaEnergy > alphaK*npts
					snapped = true;
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(lvl);
				optReg(lvl); // 更新iR
				lambda *= 0.5;
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			else
			{
				fails++;
				lambda *= 4;
				if(lambda > 10000) lambda = 10000;
			}

			bool quitOpt = false;
			// 迭代停止条件, 收敛/大于最大次数/失败2次以上
			// 退出迭代优化的条件是：增量过小，迭代次数超过该层设置的最大迭代次数，误差增大的次数超过2（即发散）。
			if(!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H,Hsc; Vec8f b,bsc;

				quitOpt = true;
			}


			if(quitOpt) break;
			iteration++;
		}
		latestRes = resOld;

	}

	// std::cout << "thisToNext.log =  " <<std::endl<< thisToNext.log() << std::endl;
	// std::cout << "refToNew_current.log =  " <<std::endl<< refToNew_current.log() << std::endl;
	// std::cout << "thisToNext_aff =  " <<std::endl<< thisToNext_aff.vec() << std::endl; // 0 0
	// std::cout << "refToNew_aff_current =  " <<std::endl<< refToNew_aff_current.vec() << std::endl; // 0 0
//[ ***step 6*** ] 优化后赋值位姿, 从底层计算上层点的深度
	// 在对所有层跟踪完成之后，得到最终优化结果： TODO 输出看一下为什么失败
	thisToNext = refToNew_current; // 参考帧与当前帧之间位姿
	thisToNext_aff = refToNew_aff_current; // 参考帧与当前帧之间光度系数

	//* 使用归一化积来更新高层逆深度值
	for(int i=0;i<pyrLevelsUsed-1;i++)
		propagateUp(i);



	// snapped 为true的条件为：平移大于一定值，150*150/2.5/2.5算下来大概xyz要各平移0.01米，或者单独平移0.02米
	frameID++; // 注意此处即使位移不够，帧数也会加
	if(!snapped) snappedAt=0; 

	if(snapped && snappedAt==0) // 相当于记录上一次位移足够的frameID，然后再这个frameID的基础上加5
		snappedAt = frameID;  // 位移足够的帧数

	// 此时：frameID=1；snappedAt=1；snapped的值根据是否接受了优化而决定。


    debugPlot(0,wraps);


	// 位移足够大, 再优化5帧才行
	// 所以初始化最少需要有七帧
	// 判断能否初始化的条件为
	return snapped && frameID > snappedAt+5;
	// 关于第二帧的处理已经分析完毕，由于此时return 为false，达不到可以初始化的条件，因此第二帧的处理到此结束。
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    bool needCall = false;
    for(IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if(!needCall) return;


	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl,hl);

	for(int i=0;i<wl*hl;i++)
		iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;



	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;

		if(!point->isGood)
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));

		else
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
	}


	//IOWrap::displayImage("idepth-R", &iRImg, false);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImage(&iRImg);
}


//* 计算能量函数和Hessian矩阵, 以及舒尔补, sc代表Schur
// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
// calcResAndGS()函数解析：
// trackFrame()函数优化的变量是两帧之间的相对状态（包括位姿和光度，8维），以及选取的像素点的逆深度，因此需要求解残差关于优化变量雅克比矩阵。
// 推导可以参考博客直接法光度误差导数推导。https://www.cnblogs.com/JingeTU/p/8203606.html
// calcResAndGS()函数计算了高斯牛顿方程的H，b；以及舒尔补之后Hsc，bsc。他们的构建以及推导可以参考博客DSO优化代码
// 中的Schur Complement. https://www.cnblogs.com/JingeTU/p/8297076.html
// refToNew 是参考帧与当前帧之间位姿， refToNew_aff 是参考帧与当前帧之间光度系数
Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff,
		bool plot)
{
	// 只有在程序最开始的时候执行
	std::cout << "Calling calcResAndGS" << std::endl; 
	int wl = w[lvl], hl = h[lvl];
	// 当前层图像及梯度,用法如下
	// colorNew[idx][0]  表示当前层lvl，idx位置处的像素的像素灰度值;
	// colorNew[idx][1]  表示当前层lvl，idx位置处的像素的x方向的梯度
	// colorNew[idx][2]  表示当前层lvl，idx位置处的像素的y方向的梯度
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];  
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

	//! 旋转矩阵R * 内参矩阵K_inv
	// 首先计算RKi t r2new_aff ，将第一帧图像选取的像素点投影到当前帧，并且投影时要根据像素点属于哪层金字塔选用对应层的金字塔内参，同时会删除
	// 投影位置不好的点。
	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>(); // R*K_inv
	// std::cout << "refToNew matrix = " << std::endl << refToNew.rotationMatrix() << std::endl;
	Vec3f t = refToNew.translation().cast<float>(); // 平移
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b); // 光度参数

	// 该层的相机参数
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];


	// acc9 是Hessian 矩阵， Accumulator9 类型，9维向量, 乘积获得9*9矩阵, 并做的累加器
	Accumulator11 E;  // 1*1 的累加器
	acc9.initialize(); // 初始值, 分配空间
	E.initialize();


	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl]; // points存储每一层上的点类, 是第一帧提取出来的，存储的是满足灰度梯度阈值的点
	for(int i=0;i<npts;i++)
	{
		// point 是当前点
		Pnt* point = ptsl+i;

		point->maxstep = 1e10;
		if(!point->isGood)  // 点不好
		{
			// energy 变量，energy[0]是残差的平方, energy[1]是正则化项(逆深度减一的平方)
			E.updateSingle((float)(point->energy[0])); // 累加
			point->energy_new = point->energy;
			point->isGood_new = false;
			continue;
		}

		// 注意DSO采用了像素点的pattern，因此每个像素点的残差是8维的。
        VecNRf dp0;  // 8*1矩阵, 每个点附近的残差个数为8个
        VecNRf dp1;
        VecNRf dp2;
        VecNRf dp3;
        VecNRf dp4;
        VecNRf dp5;
        VecNRf dp6;
        VecNRf dp7;
        VecNRf dd; // 光度误差对逆深度的导数
        VecNRf r;
		JbBuffer_new[i].setZero();  // 10*1 向量

		// sum over all residuals.
		// patternNum 的值是8
		bool isGood = true;
		float energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{
			// pattern的坐标偏移
			// 此处的 patternP 为论文中使用的8点的pattern，pattern内有40个可选点, 每个点2维xy，但只用了前8个点
			// dx dy 分别是第一个第二个 {0,-2},{-1,-1},{1,-1},{-2,0},{0,0},{2,0},{-1,1},{0,2}
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];

			//! Pj' = R*(X/Z, Y/Z, 1) + t/Z, 变换到新的点, 深度仍然使用Host帧的!
			Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new; 
			// 归一化坐标 Pj
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			// 像素坐标pj
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
			// dpi/pz' 
			float new_idepth = point->idepth_new/pt[2]; // 新一帧上的逆深度。这个对吗

			// 落在边缘附近，深度小于0, 不好 false，pattern8个点中的一个不好，则直接break
			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0))
			{
				isGood = false;
				break;
			}
			// 根据投影后的float型变量，在邻近四个角插值得到在新图像中的 patch 像素灰度值和插值后的xy方向梯度，
			// (输入3维，输出3维像素值 + x方向梯度 + y方向梯度)
			// 残差的构建以及energy的计算：（使用可Huber权重）
			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
			//Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

			// 插值得到参考帧上的 patch 上的像素灰度值, 输出一维像素灰度值
			// colorRef[idx][0]  表示当前层lvl，idx位置处的像素的像素灰度值;
			// colorRef[idx][1]  表示当前层lvl，idx位置处的像素的x方向的梯度
			// colorRef[idx][2]  表示当前层lvl，idx位置处的像素的y方向的梯度
			//float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
			//对第一帧图像只算0通道灰度图的双线性插值 没算梯度的插值
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);

			// 参考帧和当前帧的像素值无穷, false，pattern8个点中的一个不好，则直接break
			if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
			{
				isGood = false;
				break;
			}

			// 残差! 投影后的像素灰度值 减去 参考帧的像素灰度值，考虑了光度参数 TODO event camera
			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
			// setting_huberTH 为 Huber权重 Huber Threshold 值为9
			// 当残差小于9则权重为1，当残差大于9则权重为9除以残差，相当于残差越大权重越低， TODO event camera
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual); 
			// huberweight * (2-huberweight) = Objective Function
			// robust 权重和函数之间的关系
			energy += hw *residual*residual*(2-hw); // 累加pattern内8个点的残差作为energy


			// 关于优化变量的雅克比矩阵可以通过参考博客推导，然后可以构建雅克比矩阵，在构建时采用了SSE指令集加速计算。

			// 公式32：像素坐标 Pj 对 逆深度 di 求导，t 是平移向量
			// //! 1/Pz * (tx - u*tz), u = px/pz
			// float dxdd = (t[0]-t[2]*u)/pt[2]/point->idepth; // Todo bug 此处应该加上"/point->idepth"吧？
			// //! 1/Pz * (ty - v*tz), v = py/pz
			// float dydd = (t[1]-t[2]*v)/pt[2]/point->idepth;

			float dxdd = (t[0]-t[2]*u)/pt[2];
			float dydd = (t[1]-t[2]*v)/pt[2];

			if(hw < 1) hw = sqrtf(hw); //?? 为啥开根号, 答: 鲁棒核函数等价于加权最小二乘
			//! dxfx, dyfy
			float dxInterp = hw*hitColor[1]*fxl; // 权重*x方向梯度*fx
			float dyInterp = hw*hitColor[2]*fyl;
			//* 残差对 j(新状态) 位姿求导, 
			// 公式41：光度误差对se(3)六个量的导数 Jx21
			dp0[idx] = new_idepth*dxInterp; //! dpi/pz' * dxfx
			dp1[idx] = new_idepth*dyInterp; //! dpi/pz' * dyfy
			dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp); //! -dpi/pz' * (px'/pz'*dxfx + py'/pz'*dyfy)
			dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp; //! - px'py'/pz'^2*dxfy - (1+py'^2/pz'^2)*dyfy
			dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp; //! (1+px'^2/pz'^2)*dxfx + px'py'/pz'^2*dxfy
			dp5[idx] = -v*dxInterp + u*dyInterp; //! -py'/pz'*dxfx + px'/pz'*dyfy
			//* 残差对光度参数求导， rlR 是参考帧上插值得到的投影点的灰度值，r2new_aff是光度参数
			// 公式12：光度误差对辐射仿射变换两个参数的导数
			dp6[idx] = - hw*r2new_aff[0] * rlR; //! exp(aj-ai)*I(pi)
			// 公式10
			dp7[idx] = - hw*1;	//! 对 b 导
			//* 残差(光度误差)对 i(旧状态) 逆深度求导
			// 公式14 Jp：光度误差对逆深度的导数=公式15*公式20=公式15*公式23*公式32
			dd[idx] = dxInterp * dxdd  + dyInterp * dydd; 	//! dxfx * 1/Pz * (tx - u*tz) +　dyfy * 1/Pz * (tx - u*tz)
			r[idx] = hw*residual; //! 乘了Huber权重后的残差 res

			//* 像素误差对逆深度的导数，取模倒数
			// 1 / (公式20的平方根) 公式20是像素坐标对逆深度求导
			// 1 / (公式32*公式23)'平方根
			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();  //? 为什么这么设置
			if(maxstep < point->maxstep) point->maxstep = maxstep; // maxstep 是逆深度增加的最大步长

			// immediately compute dp*dd' and dd*dd' in JbBuffer1.
			//* 计算Hessian的第一行(列), 及Jr 关于逆深度那一行，注意此处的JbBuffer_new在for循环内，所以是对8个pattern点的累加
			// JbBuffer_new在 idx pattern 循环内，分别对每点的8个 pattern 的JTx21Jρ,JTρr21,JρJTρ进行累加.
			// 用来计算舒尔补
			JbBuffer_new[i][0] += dp0[idx]*dd[idx]; // Hessian 矩阵右上角，左下角部分，光度误差对位姿的偏导*光度误差对逆深度的偏导, Hpx21 <--> Jx21*Jp
			JbBuffer_new[i][1] += dp1[idx]*dd[idx];
			JbBuffer_new[i][2] += dp2[idx]*dd[idx];
			JbBuffer_new[i][3] += dp3[idx]*dd[idx];
			JbBuffer_new[i][4] += dp4[idx]*dd[idx];
			JbBuffer_new[i][5] += dp5[idx]*dd[idx];
			JbBuffer_new[i][6] += dp6[idx]*dd[idx];// Hessian 矩阵右上角，左下角部分，光度误差对仿射变换的偏导*光度误差对逆深度的偏导, Hpx21 <--> Jx21*Jp
			JbBuffer_new[i][7] += dp7[idx]*dd[idx];
			JbBuffer_new[i][8] += r[idx]*dd[idx]; // 残差(光度误差)*光度误差对逆深度的偏导, r*Jp
			JbBuffer_new[i][9] += dd[idx]*dd[idx]; // Hessian 矩阵左上角，Hpp，光度误差对逆深度求导的平方, Jp*Jp
		}
		
		// 如果点的pattern(其中一个像素)超出图像,像素值无穷, 或者残差大于阈值，--> 不是好的内点，使用上一帧的
		if(!isGood || energy > point->outlierTH*20)
		{
			// E 的类型为Accumulator11，只存储残差
			// energy[0]残差的平方, energy[1]正则化项(逆深度减一的平方)，注意此处的energy和point->energy的区别
			E.updateSingle((float)(point->energy[0])); // 上一帧的加进来
			point->isGood_new = false;
			point->energy_new = point->energy; //上一次的给当前次的
			continue;
		}

		// E 的类型为Accumulator11
		// 如果是好的内点，则加进能量函数
		// add into energy.
		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy;

		// Accumulator 类型的变量的更新过程：先通过 updateSSE()函数将变量存在 内部变量 SSEData 中，然后 shiftUp(false) 函数做判断来决定
		// 是否每1000进位，来把数据从 SSEData 放到 SSEData1k 和 SSEData1m ， 通过 updateSingle 函数来加多余的单独的，最后
		// 调用 finish()函数，来先shiftUp(true)，来把数据放到 SSEData1m ，然后从 SSEData1m 放到 内部变量 H 

		//! 因为使用128位相当于每次加4个数, 因此i+=4, 妙啊!
		// update Hessian matrix. 真正的Hessian矩阵!
		for(int i=0;i+3<patternNum;i+=4)
			acc9.updateSSE(
					_mm_load_ps(((float*)(&dp0))+i),  // Jx21 --> 位姿 x
					_mm_load_ps(((float*)(&dp1))+i),  // Jx21 --> 位姿 y
					_mm_load_ps(((float*)(&dp2))+i),  // Jx21 --> 位姿 z
					_mm_load_ps(((float*)(&dp3))+i),  // Jx21 --> 旋转xi_1
					_mm_load_ps(((float*)(&dp4))+i),  // Jx21 --> 旋转xi_2
					_mm_load_ps(((float*)(&dp5))+i),  // Jx21 --> 旋转xi_3
					_mm_load_ps(((float*)(&dp6))+i),  // Jx21 --> 辐射仿射变换两个参数
					_mm_load_ps(((float*)(&dp7))+i),  // Jx21 --> 辐射仿射变换两个参数
					_mm_load_ps(((float*)(&r))+i));   // r21

		// 加0, 4, 8后面多余的值, 因为SSE2是以128为单位相加, 多余的单独加
		// 先右移2位，再左移两位相当于把后两位去掉，然后从这里开始累加
		for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			acc9.updateSingle( // 此处应该只更新了大 H 矩阵右下角 Jx21*Jx21 ，和　b 的下边　Jx21 * r21
					(float)dp0[i],(float)dp1[i],(float)dp2[i],(float)dp3[i],
					(float)dp4[i],(float)dp5[i],(float)dp6[i],(float)dp7[i],
					(float)r[i]);


	}

	// E 是 Accumulator11 类型
	// 调用 finish()函数，来先shiftUp(true)，来把数据放到 SSEData1m ，然后从 SSEData1m 放到 内部变量 H 
	E.finish();
	acc9.finish();




	//????? 这是在干吗???

	// calculate alpha energy, and decide if we cap it.
	Accumulator11 EAlpha;
	EAlpha.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		// 如果点的pattern(其中一个像素)超出图像,像素值无穷, 或者残差大于阈值，--> 不是好的内点
		if(!point->isGood_new) // 点不好，用之前的
		{
			// TODO 和之前的不一样的地方在于这里更新energy[0]，不过这里为啥又更新E了呢？难道不应该是更新 EAlpha 吗？bug
			// energy[0]残差的平方, energy[1]正则化项(逆深度减一的平方)，注意此处的energy和point->energy的区别
			E.updateSingle((float)(point->energy[1])); //! 又是故意这样写的，没用的代码。  TODO 此处E.finish()已经被调用，所以不会再更新了。此处应该是EAlpha
		}
		else
		{
			// 最开始初始化都是成1
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);  //? 什么原理?
			E.updateSingle((float)(point->energy_new[1])); // TODO 此处应该是EAlpha
		}
	}
	EAlpha.finish(); //! 只是计算位移是否足够大
	// alphaW 的值 150*150， alphaK 的值 2.5*2.5, EAlpha.A 的值0， refToNew 是参考帧与当前帧之间位姿
	float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts); // 平移越大, 越容易初始化成功?

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);


	// compute alpha opt.
	float alphaOpt;
	if(alphaEnergy > alphaK*npts) // 平移大于一定值，150*150/2.5/2.5算下来大概xyz要各平移0.01米，或者单独平移0.02米
								  // 从后边可以看出，当满足这个条件时， snapped = true，trackFrame()才能返回true
	{
		alphaOpt = 0;
		alphaEnergy = alphaK*npts; // 2.5*2.5*npts
	}
	else
	{
		alphaOpt = alphaW; // 150*150
	}

	// Schur部分Hessian 初始化
	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;

		// JbBuffer_new[i][9] 对应 Hessian 矩阵左上角，Hpp，光度误差对逆深度求导的平方, Jp*Jp
		// point->lastHessian_new 是新一次迭代的协方差
		point->lastHessian_new = JbBuffer_new[i][9]; // 对逆深度 dd*dd

		//? 这又是啥??? 对逆深度的值进行加权? 深度值归一化?
		// 前面 Energy 加上了(d-1)*(d-1), 所以 dd = 1， r += (d-1)
		// alphaOpt 的值是 0 (位移足够大)或者 150*150 (位移不够大)
		// 前边对 JbBuffer_new[i][8] 的赋值 += r[idx]*dd[idx]; // 残差(光度误差)*光度误差对逆深度的偏导 r*Jp
		JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
		// JbBuffer_new[i][9] 对应 Hessian 矩阵左上角，Hpp，光度误差对逆深度求导的平方, Jp*Jp
		JbBuffer_new[i][9] += alphaOpt; // 对逆深度导数为1 // dd*dd

		if(alphaOpt==0)
		{
			// couplingWeight 的值为1。　point->idepth_new　为该点在新的一帧(当前帧)上的逆深度。point->iR　为逆深度的期望值
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

		// 对应 1/JpJpT
		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);  // 取逆是协方差，做权重,分母里多了一个1，猜测是为了防止JbBuffer_new[i][9]太小造成系统不稳定。
		//* 9做权重, 计算的是舒尔补项!
		//! dp*dd*(dd^2)^-1*dd*dp
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();


	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	// 对应"Gauss Newton 方程可以进一步写成"下边的公式，此处应该只更新了大 H 矩阵右下角 Jx21*Jx21 ，和　b 的下边　Jx21 * r21
	H_out = acc9.H.topLeftCorner<8,8>();// / acc9.num;  		!dp^T*dp
	b_out = acc9.H.topRightCorner<8,1>();// / acc9.num; 		!dp^T*r 
	// 对应"Schur Complement 消除"下边的公式，此处应该只更新了大 H 矩阵右下角 的被减数部分，和　b 的下边的被减数部分
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num; 	!(dp*dd)^T*(dd*dd)^-1*(dd*dp)
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;	!(dp*dd)^T*(dd*dd)^-1*(dp^T*r) // 取得是　r*Jp * Jx21*Jp/(Jp*Jp)

	//??? 啥意思
	// t*t*ntps
	// 给 t 对应的Hessian, 对角线加上一个数, b也加上
	// alphaOpt 的值是 0 (位移足够大)或者 150*150 (位移不够大)
	H_out(0,0) += alphaOpt*npts;
	H_out(1,1) += alphaOpt*npts;
	H_out(2,2) += alphaOpt*npts;

	Vec3f tlog = refToNew.log().head<3>().cast<float>(); // 李代数, 平移部分 (上一次的位姿值)
	b_out[0] += tlog[0]*alphaOpt*npts;
	b_out[1] += tlog[1]*alphaOpt*npts;
	b_out[2] += tlog[2]*alphaOpt*npts;



	// E.A 的值在 finish() 函数中更新 (只有 Accumulator11 类型的变量，在finish()函数中更新A的值
	// E.A 存储的能量值(也就是pattern点的残差和，残差由当前帧和参考帧的灰度值相减得到)； alphaEnergy 存储平移的能量值，相当于平移越大, 越容易初始化成功； E.num存储累加的点的个数
	return Vec3f(E.A, alphaEnergy ,E.num);
}

float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
	//	float factori = 1.0f/factor;
	//	float factori2 = factori*factori;
	//
	//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	//	{
	//		int npts = numPoints[lvl];
	//		Pnt* ptsl = points[lvl];
	//		for(int i=0;i<npts;i++)
	//		{
	//			ptsl[i].iR *= factor;
	//			ptsl[i].idepth_new *= factor;
	//			ptsl[i].lastHessian *= factori2;
	//		}
	//	}
	//	thisToNext.translation() *= factori;

	return factor;
}

//* 计算旧的和新的逆深度与iR的差值, 返回旧的差, 新的差, 数目
//? iR到底是啥呢     答：IR是逆深度的均值，尺度收敛到IR
Vec3f CoarseInitializer::calcEC(int lvl)
{
	if(!snapped) return Vec3f(0,0,numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(!point->isGood_new) continue;
		float rOld = (point->idepth-point->iR);
		float rNew = (point->idepth_new-point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew)); // 求和

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	// couplingWeight 的值为1
	// A1m[0]存的是老点的(逆深度-逆深度均值)的平方，A1m[1]存的是新点的(逆深度-逆深度均值)的平方，num存储累加的次数
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}

//* 使用最近点来更新每个点的iR, smooth的感觉
void CoarseInitializer::optReg(int lvl)
{
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	
	//* 位移不足够则设置iR是1
	if(!snapped)
	{
		for(int i=0;i<npts;i++)
			ptsl[i].iR = 1;
		return;
	}


	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood) continue;

		float idnn[10];
		int nnn=0;
		// 获得当前点周围最近10个点, 质量好的点的iR
		for(int j=0;j<10;j++)
		{
			if(point->neighbours[j] == -1) continue;
			Pnt* other = ptsl+point->neighbours[j];
			if(!other->isGood) continue;
			idnn[nnn] = other->iR;
			nnn++;
		}

		// 与最近点中位数进行加权获得新的iR
		if(nnn > 2)
		{
			std::nth_element(idnn,idnn+nnn/2,idnn+nnn); // 获得中位数
			// regWeight 用来对逆深度的加权值, 0.8 // TODO 这个0.8有道理么？应该是为了更相信中位数的深度
			point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
		}
	}

}


//* 使用归一化积来更新高层逆深度值
//@ param: 注意传入的参数为当前的金字塔层，并没有像propagateDown中那样+1
// 参考 propagateDown 函数的注释
void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl+1<pyrLevelsUsed);
	// set idepth of target

	// 参考 propagateDown 函数的注释
	int nptss= numPoints[srcLvl];
	int nptst= numPoints[srcLvl+1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl+1];

	// set to zero.
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		parent->iR=0;
		parent->iRSumNum=0;
	}
	// 更新当前层
	for(int i=0;i<nptss;i++)
	{
		Pnt* point = ptss+i;
		if(!point->isGood) continue;

		Pnt* parent = ptst + point->parent;
		//当前层的父节点的iR值重置一次 利用当前层点的逆深度来操作
		parent->iR += point->iR * point->lastHessian; //! 均值*信息矩阵 ∑ (sigma*u)
		parent->iRSumNum += point->lastHessian;  //! 新的信息矩阵 ∑ sigma
	}

	//* 更新在上一层的parent
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		if(parent->iRSumNum > 0)
		{
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum); //! 高斯归一化积后的均值
			parent->isGood = true;
		}
	}

	optReg(srcLvl+1); // 使用附近的点来更新IR和逆深度
}

//@ 使用上层信息来初始化下层
//@ param: 注意传入的参数为当前的金字塔层+1
//@ note: 没法初始化顶层值 
void CoarseInitializer::propagateDown(int srcLvl)
{
	assert(srcLvl>0);
	// set idepth of target

	// points 存储每一层上的点类, 是第一帧提取出来的，存储的是满足灰度梯度阈值的点
	// 在对每层图像的灰度梯度选点之后，将点存储到 points[lvl]，numPoints[lvl]表示lvl层选取的符合像素梯度阈值的像素点数量。
	int nptst= numPoints[srcLvl-1]; // 当前层的点数目
	Pnt* ptss = points[srcLvl];  // 当前层+1, 上一层的点集
	Pnt* ptst = points[srcLvl-1]; // 当前层点集

	for(int i=0;i<nptst;i++)
	{
		Pnt* point = ptst+i;  // 遍历当前层的点
		// point->parent 表示 idx (x+y*w) of closest point one pyramid level above.
		Pnt* parent = ptss+point->parent;  // 找到当前点的parent

		// isGood == true 说明当前点的像素梯度达到阈值
		// 父点不是好点(灰度梯度达不到阈值)或者逆深度的Hessian, 即协方差, dd*dd 太小，continue
		if(!parent->isGood || parent->lastHessian < 0.1) continue;
		if(!point->isGood)
		{
			// 当前点不好, 父点好，则把父点的值直接给它, 并且置位good
			point->iR = point->idepth = point->idepth_new = parent->iR;
			point->isGood=true;
			point->lastHessian=0;
		}
		else // 当前点好，父点也好，则通过hessian给当前点和父点加权求得新的iR
		{
			// 通过hessian给point和parent加权求得新的iR
			// iR可以看做是深度的值, 使用的高斯归一化积, Hessian是信息矩阵，更相信当前点的权重
			float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}
	//optReg 函数使用最近点来更新每个点的iR, smooth的感觉
	//? 为什么在这里又更新了当前层的 iR, 没有更新 idepth // TODO iR idepth 到底是什么呀？
	// 感觉更多的是考虑附近点的平滑效果
	optReg(srcLvl-1); // 当前层
}

//* 低层计算高层, 像素值和梯度
void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];
		// 使用上一层得到当前层的值
		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
		// 根据像素计算梯度
		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}

// setFirst在第一帧数据中取点：setFirst(&Hcalib, fh)：每层采集密度不一样，取点策略：
// 先将图像划分成32*32的块，计算每块的梯度均值作为选点的阈值makeHists(fh)，选点时，第一次选择先把图像分成d*d的块，然后选
// 择梯度最大且大于阈值的点，如果第一次选择结束后选择的点数目未达要求，则分成2d*2d的块，以此类推

// CoarseInitializer::setFirst，计算图像的每一层内参, 再针对不同层数选择大梯度像素, 第0层比较复杂1d, 2d, 4d大小block来选择3个层次的像素选取点，
// 其它层则选出goodpoints, 作为后续第二帧匹配生成 pointHessians 和 immaturePoints 的候选点，这些点存储在 CoarseInitializer::points 中。
// 每一层点之间都有联系，在 CoarseInitializer::makeNN 中计算每个点最邻近的10个点 neighbours ，在上一层的最邻近点 parent。
// pointHessians 是成熟点，具有逆深度信息的点，能够在其他影像追踪到的点。 immaturePoints 是未成熟点，需要使用非关键帧的影像对它的逆深度进行优化，
// 在使用关键帧将它转换成 pointHessians ，并且加入到窗口优化。
//首先利用makeK()函数计算每层图像金字塔的内参，计算方法与上一篇博客中提到的一样，并且将当前帧赋给firstFrame。
void CoarseInitializer::setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian)
{
//[ ***step 1*** ] 计算图像每层的内参
	makeK(HCalib); //计算每层图像金字塔的内参，用于后续的金字塔跟踪模型
	firstFrame = newFrameHessian;

	// 然后进行像素点选取的相关类的初始实例化以及变量的初始定义。
	PixelSelector sel(w[0],h[0]); // 像素选择

	float* statusMap = new float[w[0]*h[0]];
	bool* statusMapB = new bool[w[0]*h[0]];

	float densities[] = {0.03,0.05,0.15,0.5,1}; // 不同层取得点密度
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
//[ ***step 2*** ] 针对不同层数选择大梯度像素, 第0层比较复杂1d, 2d, 4d大小block来选择3个层次的像素
		sel.currentPotential = 3; // 设置网格大小，3*3大小格
		int npts; // 选择的像素数目
		// 接下来对第一帧进行像素点选取，利用for循环对每一层进行选点。变量 npts 表示选点数量，通过数组 statusMap 和 statusMapB 中的对应位置的值表示
		// 该位置的像素点是否被选取。
		if(lvl == 0) // 第0层提取特征像素 // 第0层(原始图像)
			// makeMaps 函数先选取像素梯度较低的前百分之五十像素点，然后在select 函数中计算梯度高于该阈值2倍的像素点，相当于还是提取梯度大的呗
			npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);
		else  // makePixelStatus 函数在其它层则选出goodpoints。即对于高层(0层以上)选择x y xy yx方向梯度最大的位置点，并标记在 statusMapB 变量上
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);


		// 在对每层图像选点之后，将点存储起来，numPoints[lvl]表示lvl层选取的像素点数量。
		// 如果点非空, 则释放空间, 创建新的
		// points 是每一层上的点类, 是第一帧提取出来的，存储的是满足灰度梯度阈值的点
		if(points[lvl] != 0) delete[] points[lvl];
		points[lvl] = new Pnt[npts]; // lvl 表示金字塔层数

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl]; // 每一层的图像大小
		Pnt* pl = points[lvl];  // 每一层上的点 //pl为lvl层的首地址，与makeImages()函数中类似
		int nl = 0;
		// 要留出pattern的空间, 2 border
//[ ***step 3*** ] 在上述函数 makeMaps 和 makePixelStatus 选出的像素中, 添加点信息到 points[lvl]
		// patternPadding 默认是2，相当于去除图片四周像素为2的边界
		// 遍历当前层的整个像素点
		for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
		for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
		{
			//if(x==2) printf("y=%d!\n",y);
			// 如果是被选中的像素
			if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0))
			{
				//assert(patternNum==9);
				// 选取的像素点相关值的初始化，nl为像素点的ID
				pl[nl].u = x+0.1;   //? 加0.1干啥
				pl[nl].v = y+0.1;
				pl[nl].idepth = 1; // 该点对应参考帧的逆深度
				pl[nl].iR = 1; // 逆深度的期望值
				pl[nl].isGood=true; // isGood == true 说明当前点的像素梯度达到阈值
				pl[nl].energy.setZero(); // [0]残差的平方, [1]正则化项(逆深度减一的平方)
				pl[nl].lastHessian=0; // 逆深度的Hessian, 即协方差, dd*dd
				pl[nl].lastHessian_new=0; // 新一次迭代的协方差
				pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];

				Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl]; // 该像素梯度
				float sumGrad2=0;
				// 计算pattern内像素梯度和. patternNum 的值为 8
				for(int idx=0;idx<patternNum;idx++)
				{
					// 此处的 patternP 为论文中使用的8点的pattern，pattern内有40个可选点, 每个点2维xy，但只用了前8个点
					// TODO 事件相机需要测试一下不同的pattern的差别
					// dx dy 分别是第一个第二个 {0,-2},{-1,-1},{1,-1},{-2,0},{0,0},{2,0},{-1,1},{0,2}
					int dx = patternP[idx][0]; // pattern 的偏移
					int dy = patternP[idx][1];
					// cpt已经是该点的像素的梯度，所以加上pattern的八个点来计算，x方向直接加就行，但是y方向要乘上当前层的宽度
					float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
					sumGrad2 += absgrad; // 累加灰度值
				}
				// TODO 此处的 sumGrad2 计算之后被原作者给注释掉了，没有使用，而是直接手动设置了阈值 8个点*x方向12*y方向12，这个值对事件相机会不会太小？

				// 把以下两个变量的定义复制过来方便理解
				// /* Outlier Threshold on photometric energy */
				// float setting_outlierTH = 12*12;					// higher -> less strict
				// float setting_outlierTHSumComponent = 50*50; 		// higher -> less strong gradient-based reweighting .

			// float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
			// pl[nl].outlierTH = patternNum*gth*gth;
				// patternNum 的值是8，setting_outlierTH 的值是12*12，像是手动设置
				//! 外点的阈值与pattern的大小有关, 一个像素是12*12
				//? 这个阈值怎么确定的...
				pl[nl].outlierTH = patternNum*setting_outlierTH;



				nl++;
				assert(nl <= npts);
			}
		}

		// 在对每层图像选点之后，将点存储起来，numPoints[lvl]表示lvl层选取的像素点数量。
		numPoints[lvl]=nl; // 每一层选出来的符合梯度阈值的数目,  去掉了一些边界上的点
	}
	delete[] statusMap;
	delete[] statusMapB;
//[ ***step 4*** ] 计算点的最近邻和父点
	// 选点之后，利用makeNN()函数建立KDtree。
	// TODO 没看明白
	makeNN();

	// 参数初始化
	// 设置一些变量的值， thisToNext 表示当前帧到下一帧的位姿变换， snapped frameID snappedAt 这三个变量
	// 在后续判断是否跟踪了足够多的图像帧能够初始化时用到。
	thisToNext=SE3();
	snapped = false;
	frameID = snappedAt = 0;

	// dGrads是啥？似乎代码中没有用到
	for(int i=0;i<pyrLevelsUsed;i++)
		dGrads[i].setZero();

}

//@ 重置点的energy, idepth_new参数
void CoarseInitializer::resetPoints(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		// 重置
		pts[i].energy.setZero();
		pts[i].idepth_new = pts[i].idepth; // TODO check where set value to idepth. Ans: In setFirst, Pnt* pl = points[lvl];
		// 如果是最顶层, 并且isGood是false(当前点的像素梯度没达到阈值)，则使用周围点平均值来重置
		// 如果不是最顶层，只执行上边两步
		if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)
		{
			float snd=0, sn=0;
			for(int n = 0;n<10;n++)
			{
				// 如果周围邻居点的像素梯度也没有达到阈值，则continue
				if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
				snd += pts[pts[i].neighbours[n]].iR; // snd --> summary of neighbours' depth
				sn += 1;
			}

			if(sn > 0) // 如果周围点像素梯度达到阈值，并且是好点
			{
				// 将当前坏点变好，对逆深度期望值、逆深度、新逆深度取周围点的平均
				pts[i].isGood=true;
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
			}
		}
	}
}

//* 求出状态增量后, 计算被边缘化掉的逆深度, 更新逆深度
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) continue;

		//! dd*r + (dp*dd)^T*delta_p 
		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		//! dd * delta_d = dd*r - (dp*dd)^T*delta_p = b 
		//! delta_d = b * dd^-1
		float step = - b * JbBuffer[i][9] / (1+lambda);


		float maxstep = maxPixelStep*pts[i].maxstep; // 逆深度最大只能增加这些
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		// 更新得到新的逆深度
		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50; // max depth is 50?
		pts[i].idepth_new = newIdepth; // 此处的pts[i].idepth_new并不是存储正确逆深度的变量，只是先存储下来作为临时变量，如果accept才赋值给下边的pts[i].idepth
	}

}

//* 新的值赋值给旧的 (能量, 点状态, 逆深度, hessian)
void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new; // 此处的pts[i].idepth才是存储正确逆深度的变量
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

//@ 计算每个金字塔层的相机参数
// That strange "0.5" offset: Internally, DSO uses the convention that the pixel at integer position (1,1) in the image, 
// i.e. the pixel in the second row and second column, contains the integral over the continuous image function from
// (0.5,0.5) to (1.5,1.5), i.e., approximates a "point-sample" of the continuous image functions at (1.0, 1.0). 
// In turn, there seems to be no unifying convention across calibration toolboxes whether the pixel at integer
// position (1,1) contains the integral over (0.5,0.5) to (1.5,1.5), or the integral over (1,1) to (2,2). The above 
// conversion assumes that the given calibration in the calibration file uses the latter convention, and thus applies
// the -0.5 correction. Note that this also is taken into account when creating the scale-pyramid (see globalCalib.cpp).
void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();
	// 求各层的K参数
	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		//* 0.5 offset 看README是设定0.5到1.5之间积分表示1的像素值？
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}
	// 求K_inverse参数
	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}



//@ 生成每一层点的KDTree, 并用其找到邻近点集和父点 
// makeNN()函数解析：
// 每一金字塔层选取的像素点构成一个KD数，为每层的点找到最近邻的10个点。pts[i].neighbours[myidx]=ret_index[k];
// 并且会在上一层找到该层像素点的parent，（最高层除外）pts[i].parent = ret_index[0];。用于后续提供初值，加速优化。
void CoarseInitializer::makeNN()
{
	const float NNDistFactor=0.05;
	// 第一个参数为 distance, 第二个是 datasetadaptor, 第三个是维数
	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS]; // 每层建立一个点云
	KDTree* indexes[PYR_LEVELS]; // 点云建立KDtree
	//* 每层建立一个KDTree索引二维点云
	// 到底用了几层 pyrLevelsUsed TODO
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		// points 是每一层上的点类, 是第一帧提取出来的，存储的是满足灰度梯度阈值的点
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]); // 二维点点云
		// 参数: 维度, 点数据, 叶节点中最大的点数(越大build快, query慢)
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];  // 搜索到的临近点
		float ret_dist[nn]; // 搜索到点的距离
		// 搜索结果, 最近的nn个和1个
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u,pts[i].v); // 当前点
			// 使用建立的KDtree, 来查询最近邻
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			//* 给每个点的neighbours赋值
			for(int k=0;k<nn;k++)
			{
				pts[i].neighbours[myidx]=ret_index[k]; // 最近的索引
				float df = expf(-ret_dist[k]*NNDistFactor); // 距离使用指数形式
				sumDF += df; // 距离和
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			// 对距离进行归10化,,,,,
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;

			//* 高一层的图像中找到该点的父节点
			if(lvl < pyrLevelsUsed-1 )
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt*0.5f-Vec2f(0.25f,0.25f); // 换算到高一层
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0]; // 父节点
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor); // 到父节点的距离(在高层中)

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			else  // 最高层没有父节点
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}



	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}

