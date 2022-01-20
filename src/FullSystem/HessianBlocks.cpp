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


 
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso
{

//@ 从ImmaturePoint构造函数, 不成熟点变地图点
PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib)
{
	instanceCounter++;
	host = rawPoint->host; // 主帧
	hasDepthPrior=false;

	idepth_hessian=0;
	maxRelBaseline=0;
	numGoodResiduals=0;

	// set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;
	assert(std::isfinite(rawPoint->idepth_max));
	//idepth_init = rawPoint->idepth_GT;

	my_type = rawPoint->my_type;  //似乎是显示用的

	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5); //深度均值
	setPointStatus(PointHessian::INACTIVE);

	int n = patternNum;
	memcpy(color, rawPoint->color, sizeof(float)*n);// 一个点对应8个像素
	memcpy(weights, rawPoint->weights, sizeof(float)*n);
	energyTH = rawPoint->energyTH;

	efPoint=0; // 指针=0


}

//@ 释放residual
void PointHessian::release()
{
	for(unsigned int i=0;i<residuals.size();i++) delete residuals[i];
	residuals.clear();
}

//@ 设置固定线性化点位置的状态
// 零空间附近的微小扰动，正负加减，1.0001的乘除
//TODO 后面求nullspaces地方没看懂, 回头再看<2019.09.18> 数学原理是啥?
void FrameHessian::setStateZero(const Vec10 &state_zero) // 此处传进来的state是0
{
	//! 前六维位姿必须是0
	assert(state_zero.head<6>().squaredNorm() < 1e-20);

	this->state_zero = state_zero;

	//! 感觉这个nullspaces_pose就是 Adj_T
	//! Exp(Adj_T*zeta)=T*Exp(zeta)*T^{-1}
	// 全局转为局部的，左乘边右乘
	//! T_c_w * delta_T_g * T_c_w_inv = delta_T_l
	//TODO 这个是数值求导的方法么???
	for(int i=0;i<6;i++)
	{
		// eps 是 espilon的缩写，表示一个很小的值，用于微小的扰动
		Vec6 eps; eps.setZero(); eps[i] = 1e-3;
		SE3 EepsP = Sophus::SE3::exp(eps); // 正的微小的扰动
		SE3 EepsM = Sophus::SE3::exp(-eps); // 负的微小的扰动
		// get_worldToCam_evalPT()得到在估计的相机位姿 worldToCam_evalPT，evalPT 可能是 evaluated pose and translation
		SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
		SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
		nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3); // 零空间附近的正微小扰动
	}
	//nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
	//nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

	//? rethink
	// scale change
	// 得到在估计的相机位姿 worldToCam_evalPT, 其传进来的值为 fh->shell->camToWorld.inverse()
	SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_P_x0.translation() *= 1.00001; // 这是在干嘛？加一个微小扰动么？乘以一个微小扰动
	w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
	SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_M_x0.translation() /= 1.00001; // 除以一个微小扰动
	w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
	nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3); // 零空间附近的正微小扰动


	nullspaces_affine.setZero();
	nullspaces_affine.topLeftCorner<2,1>()  = Vec2(1,0);
	assert(ab_exposure > 0);
	nullspaces_affine.topRightCorner<2,1>() = Vec2(0, expf(aff_g2l_0().a)*ab_exposure);
};



void FrameHessian::release()
{
	// DELETE POINT
	// DELETE RESIDUAL
	for(unsigned int i=0;i<pointHessians.size();i++) delete pointHessians[i];
	for(unsigned int i=0;i<pointHessiansMarginalized.size();i++) delete pointHessiansMarginalized[i];
	for(unsigned int i=0;i<pointHessiansOut.size();i++) delete pointHessiansOut[i];
	for(unsigned int i=0;i<immaturePoints.size();i++) delete immaturePoints[i];


	pointHessians.clear();
	pointHessiansMarginalized.clear();
	pointHessiansOut.clear();
	immaturePoints.clear();
}

//* 构建图像金字塔，并计算各层金字塔图像的像素值和梯度
// makeImages()函数解析：
// 首先对每层金字塔初始化两个数组，数组的类型为Eigen::Vector3f和float；其中 dIp[i] absSquaredGrad[i]是金字塔第i层的数组的首地址。
// dIp[i]数组的元素为三维向量，分别代表像素灰度值，x方向的梯度，y方向的梯度。absSquaredGrad[i]数组的元素代表两个方向的梯度平方和。
// 对这两个数据的调用方式如下：（在候选点选取的时候会用到此处构建的两个数组，在这里先介绍一下，能够帮助理解这两个数组是如何构建的。）
void FrameHessian::makeImages(float* color, CalibHessian* HCalib)
{
	// 每一层创建图像值, 和图像梯度的存储空间
	// 1. dIp 每一层图像的辐射值、x 方向梯度、y 方向梯度；2. dI 指向 dIp[0] 也就是原始图像的信息；3. absSquaredGrad 存储 xy 方向梯度值的平方和。
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		// 首先对每层金字塔初始化两个数组，数组的类型为Eigen::Vector3f和float
		// dIp[i] 和 absSquaredGrad[i] 是金字塔第i层的数组的首地址
		// dIp[i]数组的元素为三维向量，分别代表像素灰度值，x方向的梯度，y方向的梯度。
		dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
		// absSquaredGrad[i]数组的元素代表两个方向的梯度平方和。
		absSquaredGrad[i] = new float[wG[i]*hG[i]];
	}

/*	
	// 帮助理解
	// 对dIp和absSquaredGrad这两个数据的调用方式如下：（在候选点选取的时候会用到此处构建的两个数组，在这里先介绍一下，能够帮助理解这两个数组是如何构建的。）
	dI=dIp[0];  获取金字塔第0层，若要获取其他层，修改中括号里面即可；
	dI[idx][0]  表示图像金字塔第0层，idx位置处的像素的像素灰度值;(这是因为DSO中存储图像像素值都是采用一维数组来表示，类似于opencv里面的data数组。)
	dI[idx][1]  表示图像金字塔第0层，idx位置处的像素的x方向的梯度
	dI[idx][2]  表示图像金字塔第0层，idx位置处的像素的y方向的梯度
	abs=absSquaredGrad[1]; ///获取金字塔第1层，若要获取其他层，修改中括号里面即可；
	abs[idx]   表示图像金字塔第1层， idx 位置处的像素x,y方向的梯度平方和
	此两个数据的构建：主要内容是图像金字塔是如何产生以及梯度的求取（在上一篇博客中运行前的准备介绍了如何确定使用的金字塔层数。）
*/

	dI = dIp[0]; // 原来他们指向同一个地方


	// make d0
	int w=wG[0]; // 零层weight
	int h=hG[0]; // 零层height
	for(int i=0;i<w*h;i++)
		dI[i][0] = color[i]; // 第0层没有缩放，所以直接使用color赋值

	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = wG[lvl], hl = hG[lvl]; // 该层图像大小，每层在上一层的基础上除以2
		Eigen::Vector3f* dI_l = dIp[lvl]; // dI_l 指向当前层首个数据
										  // dI_l[idx][0]  表示当前层lvl，idx位置处的像素的像素灰度值;
										  // dI_l[idx][1]  表示当前层lvl，idx位置处的像素的x方向的梯度
										  // dI_l[idx][2]  表示当前层lvl，idx位置处的像素的y方向的梯度

		float* dabs_l = absSquaredGrad[lvl]; // dabs_l 指向当前层首个数据
		if(lvl>0) // 如果不是最底层，则利用下层的灰度值来四合一出当前层的灰度值
		{
			int lvlm1 = lvl-1;
			int wlm1 = wG[lvlm1]; // 当前层的上一层的列数，
			Eigen::Vector3f* dI_lm = dIp[lvlm1]; // dI_lm 表示当前层的下一层图像金字塔的首地址


			// 像素4合1, 生成金字塔，即上层金字塔图像的像素值是由下层图像的4个像素值均匀采样得到的
			// 下述代码中 dI_l 表示当前层金字塔图像首地址， dI_lm 表示下层图像金字塔的首地址。
			// dI_lm[]数组里边用了"乘以2"来计算index，是因为金字塔的每层的大小在之前一层的基础上除以2
			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
					dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] + // 此处dI_lm[][0]的0表示取vec3中的第一个元素，也就是像素灰度值
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
				}
		}

		for(int idx=wl;idx < wl*(hl-1);idx++) // idx等于wl而不是0，说明从第二行开始，因为下边算梯度需要计算 idx-wl，其实
		{
			// 此种方法对图像左右边界处的梯度计算有误呀，但可能影响不大
			// 梯度的求取：利用前后两个像素的差值作为x方向的梯度，利用上下两个像素的差值作为y方向的梯度，注意会跳过边缘像素点的梯度计算。
			float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]); // 当前点右边-当前点左边
			float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]); // 当前点下边-当前点上边


			if(!std::isfinite(dx)) dx=0; // 如果梯度无限大，dx置零
			if(!std::isfinite(dy)) dy=0;

			dI_l[idx][1] = dx; // 梯度
			dI_l[idx][2] = dy;


			dabs_l[idx] = dx*dx+dy*dy; // 梯度平方

			if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
			{
				//! 乘上响应函数, 变换回正常的颜色, 因为光度矫正时 I = G^-1(I) / V(x)
				float gw = HCalib->getBGradOnly((float)(dI_l[idx][0])); 
				// 最后会根据参数设置，对计算的两个方向的梯度平方和乘以一个权重。会用到函数setGammaFunction()里面计算的Hcalib.B[i]。
				dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
			}
		}
	}
}

//@ 计算优化前和优化后的相对位姿, 相对光度变化, 及中间变量
void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib )
{
	this->host = host;    // 这个是赋值, 计数会增加, 不是拷贝
	this->target = target;
	
	//? 实在不懂leftToleft_0这个名字怎么个含义
	// 优化前host target间位姿变换
	SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
	PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
	PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();


	// 优化后host到target间位姿变换
	SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
	PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
	PRE_tTll = (leftToLeft.translation()).cast<float>();
	distanceLL = leftToLeft.translation().norm();

	// 乘上内参, 中间量?
	Mat33f K = Mat33f::Zero();
	K(0,0) = HCalib->fxl();
	K(1,1) = HCalib->fyl();
	K(0,2) = HCalib->cxl();
	K(1,2) = HCalib->cyl();
	K(2,2) = 1;
	PRE_KRKiTll = K * PRE_RTll * K.inverse();
	PRE_RKiTll = PRE_RTll * K.inverse();
	PRE_KtTll = K * PRE_tTll;

	// 光度仿射值
	PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
	PRE_b0_mode = host->aff_g2l_0().b;
}

}

