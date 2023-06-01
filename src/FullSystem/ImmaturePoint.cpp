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



#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"
#include "FullSystem/ResidualProjections.h"

namespace dso
{
//! 这里u_ v_ 是加了0.5的
ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib)
: u(u_), v(v_), host(host_), my_type(type), idepth_min(0), idepth_max(NAN), lastTraceStatus(IPS_UNINITIALIZED)
{

	gradH.setZero();

	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		// 由于+0.5导致积分, 插值得到值3个 [像素值, dx, dy]
		// ptc[0] 像素灰度值, ptc[1] x方向灰度梯度, ptc[2] y方向灰度梯度
        Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u+dx, v+dy,wG[0]); 



		color[idx] = ptc[0];
		if(!std::isfinite(color[idx])) {energyTH=NAN; return;}

		// gradH 是8个patternP点的灰度梯度矩阵[dx*2, dxdy; dydx, dy^2]
		gradH += ptc.tail<2>()  * ptc.tail<2>().transpose();
		//! 点的权重 c^2 / ( c^2 + ||grad||^2 )
		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}

	energyTH = patternNum*setting_outlierTH;
	energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

	idepth_GT=0;
	quality=10000;
}

ImmaturePoint::~ImmaturePoint()
{
}



/* 
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
 //@ 使用深度滤波对未成熟点进行深度估计
// traceOn()函数解析：
// （1）利用 idepth_min 计算出未成熟点在当前帧的投影位置，得到（ uMin，vMin ），对投影位置进行判断，不满足条件的设置 ImmaturePointStatus::IPS_OOB;
// （2）若定义了 idepth_max ，则利用 idepth_max 计算出未成熟点在当前帧的投影位置，得到（ uMax，vMax ），对投影位置进行判断，不满足条件的设置 ImmaturePointStatus::IPS_OOB;
// （3）若未定义 idepth_max ，则取 idepth_max 为0.01，计算出极线方向，设置步长为 dist = maxPixSearch;，则可以通过计算得到（ uMax ， vMax ）。
//	   判断 uMax 和 vMax ，不满足条件的设置 ImmaturePointStatus::IPS_OOB;。
// （4）利用前三步计算的结果确定极线搜索的方向，然后通过调整步长（次数为numSteps），选取其中最好的结果作为初值用于后续的高斯牛顿优化。
// （5）注意此处优化的变量是步长，迭代优化的过程与其他优化一样，通过setting_trace_GNIterations和setting_trace_GNThreshold确定何时退出迭代。
// （6）优化结束之后，返回未成熟点的跟踪状态。
// hostToFrame_* 的旋转矩阵是 hostToNew (即host帧到最新帧)
ImmaturePointStatus ImmaturePoint::traceOn(FrameHessian* frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f& hostToFrame_affine, 
											CalibHessian* HCalib, bool debugPrint)
{
	if(lastTraceStatus == ImmaturePointStatus::IPS_OOB) return lastTraceStatus;


	debugPrint = false;//rand()%100==0;
	float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;  // 极限搜索的最大长度 = resolution * 0.027 = (640+480)*0.027 = 8294.4
	// std::cout << "wG hG = " << wG[0] << " " << hG[0] << std::endl;

	if(debugPrint)
		printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n",
				u,v,
				host->shell->id, frame->shell->id,
				idepth_min, idepth_max,
				hostToFrame_Kt[0],hostToFrame_Kt[1],hostToFrame_Kt[2]);

	//	const float stepsize = 1.0;				// stepsize for initial discrete search.
	//	const int GNIterations = 3;				// max # GN iterations
	//	const float GNThreshold = 0.1;				// GN stop after this stepsize.
	//	const float extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
	//	const float slackInterval = 0.8;			// if pixel-interval is smaller than this, leave it be.
	//	const float minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.
	
	// ============== project min and max. return if one of them is OOB ===================
//[ ***step 1*** ] 计算出来搜索的上下限, 对应idepth_max, idepth_min
	// （1）利用idepth_min计算出未成熟点在当前帧的投影位置，得到（uMin，vMin），对投影位置进行判断，不满足条件的设置ImmaturePointStatus::IPS_OOB;
	// u v 是host里的像素坐标， hostToFrame_KRKi 是 host到最新帧的像素坐标系之间的坐标变换
	Vec3f pr = hostToFrame_KRKi * Vec3f(u, v, 1); // 投影到最新帧的像素坐标系的值。此时pf[2]的值不是1，在1附近
	// std::cout << "pr = " << pr.transpose() << std::endl;
	// std::cout << "idepth_min = " << idepth_min << std::endl;
	Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min; // idepth_min 是逆深度范围的最小值，相当于假设逆深度最小的时候host里的像素坐标投影到最新帧的像素坐标系的值
													// 为什么不用预估的值呢？
	float uMin = ptpMin[0] / ptpMin[2]; // 从此处开始搜索
	float vMin = ptpMin[1] / ptpMin[2];

	// 如果超出图像范围则设为 OOB
	if(!(uMin > 4 && vMin > 4 && uMin < wG[0]-5 && vMin < hG[0]-5))
	{
		if(debugPrint) printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n",
				u,v,uMin, vMin,  ptpMin[2], idepth_min, idepth_max);
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	float dist;
	float uMax;
	float vMax;
	Vec3f ptpMax;
	// （2）若定义了 idepth_max ，则利用 idepth_max 计算出未成熟点在当前帧的投影位置，得到（ uMax ， vMax ），对投影位置进行判断，不满足条件的设置 ImmaturePointStatus::IPS_OOB;
	// 默认没有定义 idepth_max
	if(std::isfinite(idepth_max))
	{
		ptpMax = pr + hostToFrame_Kt*idepth_max; // idepth_max 是逆深度范围的最大值，相当于假设逆深度最大的时候host里的像素坐标投影到最新帧的像素坐标系的值
		uMax = ptpMax[0] / ptpMax[2]; // 在此处结束搜索
		vMax = ptpMax[1] / ptpMax[2];


		if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
		{
			if(debugPrint) printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}



		// ============== check their distance. everything below 2px is OK (-> skip). ===================
		dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax); // distance in pixel coordinate // 像素坐标系下逆深度最大和最小时候的xy的距离平方和
		dist = sqrtf(dist);
		//* 搜索的范围太小
		if(dist < setting_trace_slackInterval) // if pixel-interval is smaller than this, leave it be. 不管它
		{
			if(debugPrint)
				printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

			// lastTraceUV 是上一次搜索得到的位置， lastTracePixelInterval 是上一次的搜索范围长度
			lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;  // 直接设为中值
			lastTracePixelInterval=dist;
			return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED; //跳过
		}
		assert(dist>0);
	}
	// （3）若未定义 idepth_max ，则取 idepth_max 为0.01，计算出极线方向，设置步长为 dist = maxPixSearch;，则可以通过计算得到（ uMax ， vMax ）。
	//	   判断 uMax 和 vMax ，不满足条件的设置 ImmaturePointStatus::IPS_OOB;。
	else // 如果是第一次更新，那么最大搜索长度固定为maxPixSearch
	{
		//* 上限无穷大, 则设为最大值
		dist = maxPixSearch; // maxPixSearch的值为0.027*(w+h) 对于(640+480)*0.027 = 30.24

		// project to arbitrary depth to get direction.
		// 任意投影到较为合理的最大逆深度，深度100，只为获得极线方向
		ptpMax = pr + hostToFrame_Kt*0.01;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		// direction. 此处的direction是逆深度最大和最小的时候host里的像素坐标投影到最新帧的像素坐标系的值之间的方向
		float dx = uMax-uMin;
		float dy = vMax-vMin;
		float d = 1.0f / sqrtf(dx*dx+dy*dy); // 1/distance 作为一步的长度

		//* 根据比例得到最大值
		// set to [setting_maxPixSearch].
		uMax = uMin + dist*dx*d; // uMin + maxPixSearch*(uMax-uMin)*1.0f / sqrtf(dx*dx+dy*dy)
		vMax = vMin + dist*dy*d;

		// may still be out!
		if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
		{
			if(debugPrint) printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
			// std::cout << "dist = " << dist << std::endl;
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}
		assert(dist>0);
	}

	//? 为什么是这个值呢??? 0.75 - 1.5 
	// 这个值是两个帧上深度的比值, 它的变化太大就是前后尺度变化太大了
	// set OOB if scale change too big.
	// 如果idepth_min>=0并且ptpMin<=0.75或>=1.5，才进入if
	if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5)))
	{
		if(debugPrint) printf("OOB SCALE %f %f %f!\n", uMax, vMax,  ptpMin[2]);
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

//[ ***step 2*** ] 计算误差大小(图像梯度和极线夹角大小), 夹角大, 小的几何误差会有很大影响
	// 计算像素匹配不确定度， Vec2f(dx, dy)表示极线方向，Vec2f(dy, -dx)表示极线的垂直方向，gradH表示的是特征点周围8邻域像素hessian梯度求和。
	// 按照前面理论分析，当极线的方向和特征点梯度的方向垂直时，匹配误差较大，当极线方向和特征点梯度平行时，匹配误差最小；因此代码中的a可以表示极线与梯度的点乘再求平方，
	// 即前面公式中的. 假设极线与梯度平行，此时a特别大，b相对较小，因此（a+b）/a接近于1，那么在最优的情况下每一个像素会产生0.4个像素的误差；此时的0.4可以看做基础噪声;
	// 如果极线和梯度垂直，此时a非常小，而b非常大，所以误差就会明显增加
	// ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
	float dx = setting_trace_stepsize*(uMax-uMin); // setting_trace_stepsize 是 stepsize for initial discrete search. 值为1.0
	float dy = setting_trace_stepsize*(vMax-vMin);

	// Vec2f(dx,dy)是极线方向，Vec2f(dy, -dx)表示极线的垂直方向，gradH表示的是特征点周围8邻域像素hessian梯度求和 [dx*2, dxdy; dydx, dy^2]
	//! (dIx*dx + dIy*dy)^2
	float a = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy)); 
	//! (dIx*dy - dIy*dx)^2
	float b = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx)); // (dx, dy)垂直方向的乘积
	// 计算的是极线搜索方向和梯度方向的夹角大小，90度则a=0, errorInPixel变大；平行时候b=0
	float errorInPixel = 0.2f + 0.2f * (a+b) / a; // 没有使用LSD的方法, 估计是能有效防止位移小的情况

	//* errorInPixel大说明极线搜索方向和梯度方向垂直, 这时误差会很大, 视为bad
	if(errorInPixel*setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max))
	{
		if(debugPrint)
			printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
		lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
		lastTracePixelInterval=dist;
		return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
	}

	if(errorInPixel >10) errorInPixel=10;



	// ============== do the discrete search ===================
//[ ***step 3*** ] 在极线上找到最小的光度误差的位置, 并计算和第二好的比值作为质量
	// （4）利用前（3）步计算的结果确定极线搜索的方向，然后通过调整步长（次数为 numSteps ），选取其中最好的结果作为初值用于后续的高斯牛顿优化。
	dx /= dist; // cos // 此处的dist应该为16.362，现在的dx和dy是一个小步长
	dy /= dist;	// sin

	if(debugPrint)
		printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
				u,v,
				host->shell->id, frame->shell->id,
				idepth_min, uMin, vMin,
				idepth_max, uMax, vMax,
				errorInPixel
				);


	if(dist>maxPixSearch) // > 30.24 ?
	{
		uMax = uMin + maxPixSearch*dx;
		vMax = vMin + maxPixSearch*dy;
		dist = maxPixSearch;
	}

	// 完成上述的处理后就是具体的在极线上搜索匹配点的工作，在极线上以1个像素为步长进行搜索。每到一个像素位置计算其8邻域图像块的灰度残差，
	// 并将最小残差和最小残差对应的像素位置进行记录。最多进行100个步长的搜索。注意此处得到的是粗精度的：1 pixel

	int numSteps = 1.9999f + dist / setting_trace_stepsize; // 步数
	Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>();

	float randShift = uMin*1000-floorf(uMin*1000); // 	取小数点后面的做随机数??
	float ptx = uMin-randShift*dx; // 这个得到的是uMin左上角的一个值，相当于从uvMin的前边一点开始算
	float pty = vMin-randShift*dy;

	//* pattern在新的帧上的偏移量
	Vec2f rotatetPattern[MAX_RES_PER_POINT]; // MAX_RES_PER_POINT的值为8
	for(int idx=0;idx<patternNum;idx++)
		rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);



	// 这个判断太多了, 学习学习, 全面考虑
	if(!std::isfinite(dx) || !std::isfinite(dy))
	{
		//printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}


	//* 沿着极线搜索误差最小的位置
	float errors[100];
	float bestU=0, bestV=0, bestEnergy=1e10;
	int bestIdx=-1;
	if(numSteps >= 100) numSteps = 99;

	for(int i=0;i<numSteps;i++)
	{
		float energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{
			float hitColor = getInterpolatedElement31(frame->dI,
										(float)(ptx+rotatetPattern[idx][0]),
										(float)(pty+rotatetPattern[idx][1]),
										wG[0]);

			if(!std::isfinite(hitColor)) {energy+=1e5; continue;}
			float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);
		}

		if(debugPrint)
			printf("step %.1f %.1f (id %f): energy = %f!\n",
					ptx, pty, 0.0f, energy);


		errors[i] = energy;
		if(energy < bestEnergy) // energy变小
		{
			bestU = ptx; bestV = pty; bestEnergy = energy; bestIdx = i;
		}

		// 每次走dx/dist对应大小
		ptx+=dx; // dx是一个小步长，之前有dx /= dist
		pty+=dy;
	}

	//* 在一定的半径内找最到误差第二小的, 差的足够大, 才更好(这个常用) // 应该是在一定的半径外找误差第二小的
	// find best score outside a +-2px radius.
	float secondBest=1e10;
	for(int i=0;i<numSteps;i++) // setting_minTraceTestRadius的值是2
	{
		if((i < bestIdx-setting_minTraceTestRadius || i > bestIdx+setting_minTraceTestRadius) && errors[i] < secondBest)
			secondBest = errors[i]; // 在bestIdx附近之外的地方找，不能在附近找 TODO 合理么
	}
	float newQuality = secondBest / bestEnergy;
	if(newQuality < quality || numSteps > 10) quality = newQuality;

//[ ***step 4*** ] 在上面的最优位置进行线性搜索, 进行求精
	// 根据上面一步，已经找到了1个像素精度的最佳匹配点，接着利用高斯牛顿算法继续优化亚像素级别的最佳匹配位置。
	// ============== do GN optimization ===================
	// （5）注意此处优化的变量是步长，迭代优化的过程与其他优化一样，通过 setting_trace_GNIterations 和 setting_trace_GNThreshold 确定何时退出迭代。
	float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;
	if(setting_trace_GNIterations>0) bestEnergy = 1e5; // setting_trace_GNIterations的值是3
	int gnStepsGood=0, gnStepsBad=0;
	for(int it=0;it<setting_trace_GNIterations;it++)
	{
		float H = 1, b=0, energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{
			Vec3f hitColor = getInterpolatedElement33(frame->dI,
					(float)(bestU+rotatetPattern[idx][0]),
					(float)(bestV+rotatetPattern[idx][1]),wG[0]);

			if(!std::isfinite((float)hitColor[0])) {energy+=1e5; continue;}
			float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			float dResdDist = dx*hitColor[1] + dy*hitColor[2]; // 极线方向梯度
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			H += hw*dResdDist*dResdDist;
			b += hw*residual*dResdDist;
			energy += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
		}


		if(energy > bestEnergy) // 如果能量(残差)变大，说明优化的方向不对，减小步长
		{
			gnStepsBad++;

			// do a smaller step from old point.
			stepBack*=0.5;  		//* 减小步长再进行计算
			bestU = uBak + stepBack*dx;
			bestV = vBak + stepBack*dy;
			if(debugPrint)
				printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
						it, energy, H, b, stepBack,
						uBak, vBak, bestU, bestV);
		}
		else // 如果能量(残差)变小，说明优化的方向正确
		{
			gnStepsGood++;

			float step = -gnstepsize*b/H;
			//* 步长最大才0.5
			if(step < -0.5) step = -0.5;
			else if(step > 0.5) step=0.5;

			if(!std::isfinite(step)) step=0;

			uBak=bestU; // 备份
			vBak=bestV;
			stepBack=step;

			bestU += step*dx;
			bestV += step*dy;
			bestEnergy = energy;

			if(debugPrint)
				printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
						it, energy, H, b, step,
						uBak, vBak, bestU, bestV);
		}

		if(fabsf(stepBack) < setting_trace_GNThreshold) break;
	}


	// ============== detect energy-based outlier. ===================
	//	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU, bestV, wG[0]);
	//	float absGrad1 = getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25, bestV*0.5-0.25, wG[1]);
	//	float absGrad2 = getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375, bestV*0.25-0.375, wG[2]);
	//* 残差太大, 则设置为外点
	// setting_trace_extraSlackOnTH的值是1.2 for energy-based outlier check, be slightly more relaxed by this factor.
	if(!(bestEnergy < energyTH*setting_trace_extraSlackOnTH))
	//			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
	//		     && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
	//			 && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
	{
		if(debugPrint)
			printf("OUTLIER!\n");

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)   
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;   //? 外点还有机会变回来???
		else
			return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

//[ ***step 5*** ] 根据得到的最优位置重新计算逆深度的范围
	// 后边使用的是 idepth_min 和 idepth_max 的中值，作为此点的逆深度
	// ============== set new interval ===================
	//! u = (pr[0] + Kt[0]*idepth) / (pr[2] + Kt[2]*idepth) ==> idepth = (u*pr[2] - pr[0]) / (Kt[0] - u*Kt[2])
	//! v = (pr[1] + Kt[1]*idepth) / (pr[2] + Kt[2]*idepth) ==> idepth = (v*pr[2] - pr[1]) / (Kt[1] - v*Kt[2])
	//* 取误差最大的
	if(dx*dx>dy*dy) // dx是一个小步长
	{	// pr 是 uv经过了hostToFrame_KRKi投影之后得到的新的坐标点
		// errorInPixel 是判断极线搜索方向和梯度方向的变量，大说明极线搜索方向和梯度方向垂直, 这时误差会很大, 视为bad
		idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU-errorInPixel*dx));
		idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU+errorInPixel*dx));
	}
	else
	{
		idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV-errorInPixel*dy));
		idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV+errorInPixel*dy));
	}
	if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);
	// std::cout << "idepth min max = "<<idepth_min << "  " << idepth_max << std::endl;


	// （6）优化结束之后，返回未成熟点的跟踪状态。  没太看出来

	if(!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max<0))
	{
		//printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	lastTracePixelInterval=2*errorInPixel; 	// 搜索的范围
	lastTraceUV = Vec2f(bestU, bestV);		// 上一次得到的最优位置
	return lastTraceStatus = ImmaturePointStatus::IPS_GOOD; 	//上一次的状态
}


float ImmaturePoint::getdPixdd(
		CalibHessian *  HCalib,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth)
{
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	float drescale, u=0, v=0, new_idepth;
	float Ku, Kv;
	Vec3f KliP;

	projectPoint(this->u,this->v, idepth, 0, 0,HCalib,
			precalc->PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth);

	float dxdd = (PRE_tTll[0]-PRE_tTll[2]*u)*HCalib->fxl();
	float dydd = (PRE_tTll[1]-PRE_tTll[2]*v)*HCalib->fyl();
	return drescale*sqrtf(dxdd*dxdd + dydd*dydd);
}


float ImmaturePoint::calcResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth)
{
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
	Vec2f affLL = precalc->PRE_aff_mode;

	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;
		if(!projectPoint(this->u+patternP[idx][0], this->v+patternP[idx][1], idepth, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{return 1e10;}

		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
		if(!std::isfinite((float)hitColor[0])) {return 1e10;}
		//if(benchmarkSpecialOption==5) hitColor = (getInterpolatedElement13BiCub(tmpRes->target->I, Ku, Kv, wG[0]));

		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
	}

	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
	}
	return energyLeft;
}



//@ 计算当前点逆深度的残差, 正规方程(H和b), 残差状态
double ImmaturePoint::linearizeResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float &Hdd, float &bd,
		float idepth)
{
	if(tmpRes->state_state == ResState::OOB)
		{ tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy; }

	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	// check OOB due to scale angle change.

	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_RTll = precalc->PRE_RTll;
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	//const float * const Il = tmpRes->target->I;

	Vec2f affLL = precalc->PRE_aff_mode;

	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		if(!projectPoint(this->u,this->v, idepth, dx, dy,HCalib,
				PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}


		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

		if(!std::isfinite((float)hitColor[0])) {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}
		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);

		// depth derivatives.
		float dxInterp = hitColor[1]*HCalib->fxl();
		float dyInterp = hitColor[2]*HCalib->fyl();
		float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale); // 对逆深度的导数

		hw *= weights[idx]*weights[idx];

		Hdd += (hw*d_idepth)*d_idepth; // 对逆深度的hessian
		bd += (hw*residual)*d_idepth; // 对逆深度的Jres
	}


	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
		tmpRes->state_NewState = ResState::OUTLIER;
	}
	else
	{
		tmpRes->state_NewState = ResState::IN;
	}

	tmpRes->state_NewEnergy = energyLeft;
	return energyLeft;
}



}
