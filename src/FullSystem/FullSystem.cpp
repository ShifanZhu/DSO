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

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

namespace dso
{
// Hessian矩阵计数, 有点像 shared_ptr
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;


/********************************
 * @ function: 构造函数
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
FullSystem::FullSystem()
{

	int retstat =0;
	if(setting_logStuff)
	{
		//shell命令删除旧的文件夹, 创建新的
		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		// 打开读写log文件
		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847); // shell正常执行结束返回这么个值,填充8~15位bit, 有趣



	selectionMap = new float[wG[0]*hG[0]];

	// 比较重要的类的初始化
	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	// 以及一些变量的初始值
	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100); //5维向量都=100

	currentMinActDist=2;
	initialized=false;


	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;


	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this); // 建图线程单开
	lastRefStopID=0;



	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	// 删除new的ofstream
	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	delete[] selectionMap;

	for(FrameShell* s : allFrameHistory)
		delete s;
	for(FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

//* 设置相机响应函数
// setGammaFunction(): 将经转换后的pcalib.txt 文件的数据G[i]进行一个运算后赋值给Hcalib.B[i]。
// 其中 reader->getPhotometricGamma()获取的是经转换之后的 pcalib.txt 文件的数据，G[i]，也就是此处的 BInv.
void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);

	for(FrameShell* s : allFrameHistory)
	{
		if(!s->poseValid) continue;

		if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

		myfile << s->timestamp <<
			" " << s->camToWorld.translation().transpose()<<
			" " << s->camToWorld.so3().unit_quaternion().x()<<
			" " << s->camToWorld.so3().unit_quaternion().y()<<
			" " << s->camToWorld.so3().unit_quaternion().z()<<
			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
	}
	myfile.close();
}

//@ 使用确定的运动模型对新来的一帧进行跟踪, 得到位姿和光度参数
// 对参考帧进行跟踪
Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);



	// （1）获取参考帧的信息
	FrameHessian* lastF = coarseTracker->lastRef;  // 参考帧

	AffLight aff_last_2_l = AffLight(0,0);
//[ ***step 1*** ] 设置不同的运动状态
	// （2）设置参考帧对当前的位姿 lastF_2_fh_tries （这货是个vector），会进行一系列的
	// 假设：利用匀速模型加一系列微小旋转，如果关键帧数量只有两帧，则设置相对位姿为单位阵。
	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	printf("size: %d \n", lastF_2_fh_tries.size());
	if(allFrameHistory.size() == 2)
		for(unsigned int i=0;i<lastF_2_fh_tries.size();i++) lastF_2_fh_tries.push_back(SE3());  //? 这个size()不应该是0么
	else
	{
		FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];   // 上一帧
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];  // 大上一帧
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;  // 上一帧和大上一帧的运动
			lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;	// 参考帧到上一帧运动
			aff_last_2_l = slast->aff_g2l;
		}
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast. // 当前帧到上一帧 = 上一帧和大上一帧的

		//! 尝试不同的运动
		// get last delta-movement.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
		lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

		//! 尝试不同的旋转变动
		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
		{
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}

		if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) // 有不和法的
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}


	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);


	//! as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	//! I'll keep track of the so-far best achieved residual for each level in achievedRes. 
	//! 把到目前为止最好的残差值作为每一层的阈值
	//! If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.
	//! 粗层的能量值大, 也不继续优化了, 来节省时间
	 

	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;
	//! 逐个尝试
	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
//[ ***step 2*** ] 尝试不同的运动状态, 得到跟踪是否良好
		// （3）利用建立的一系列相对位姿利用 trackNewestCoarse() 进行跟踪，根据跟踪优化结果决定 trackingIsGood ，然后结合 lastResiduals 决定 haveOneGood 。
		// 使用满足条件的假设相对位姿作为两帧相对位姿初值。若一系列位姿假设中均不符合要求，则选取第一个。
		AffLight aff_g2l_this = aff_last_2_l;  // 上一帧的赋值当前帧
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
		
		// 根据优化结果设置返回值，传递给 trackingIsGood
		bool trackingIsGood = coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this, aff_g2l_this,
				pyrLevelsUsed-1,
				achievedRes);	// in each level has to be at least as good as the last try.
		tryIterations++;

		if(i != 0)
		{
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					i,
					i, pyrLevelsUsed-1,
					aff_g2l_this.a,aff_g2l_this.b,
					achievedRes[0],
					achievedRes[1],
					achievedRes[2],
					achievedRes[3],
					achievedRes[4],
					coarseTracker->lastResiduals[0],
					coarseTracker->lastResiduals[1],
					coarseTracker->lastResiduals[2],
					coarseTracker->lastResiduals[3],
					coarseTracker->lastResiduals[4]);
		}

//[ ***step 3*** ] 如果跟踪正常, 并且0层残差比最好的还好留下位姿, 保存最好的每一层的能量值
	// （4）根据前面的优化结果设置当前帧的相关状态：
		// do we have a new winner?
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// take over achieved res (always).
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
			{
				if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
					achievedRes[i] = coarseTracker->lastResiduals[i]; // 里面保存的是各层得到的能量值
			}
		}

//[ ***step 4*** ] 小于阈值则暂停, 并且为下次设置阈值
        if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;

	}

	if(!haveOneGood)
	{
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
		flowVecs = Vec3(0,0,0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	//! 把这次得到的最好值给下次用来当阈值
	lastCoarseRMSE = achievedRes;

//[ ***step 5*** ] 此时shell在跟踪阶段, 没人使用, 设置值
	// no lock required, as fh is not used anywhere yet.
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;


	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];  // 第一次跟踪的平均能量值

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
						<< fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure << " "
						<< fh->shell->camToWorld.log().transpose() << " "
						<< aff_g2l.a << " "
						<< aff_g2l.b << " "
						<< achievedRes[0] << " "
						<< tryIterations << "\n";
	}


	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

//@ 利用新的帧 fh 对关键帧中的ImmaturePoint进行更新
void FullSystem::traceNewCoarse(FrameHessian* fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	K(0,0) = Hcalib.fxl();
	K(1,1) = Hcalib.fyl();
	K(0,2) = Hcalib.cxl();
	K(1,2) = Hcalib.cyl();

	// 遍历关键帧
	// 遍历frameHessians，遍历所有ImmaturePoint，(此时在第八帧上提取了点，生成了ImmaturePoint)利用函数traceOn进行跟踪（又称极线搜索）。
	for(FrameHessian* host : frameHessians)		// go through all active frames
	{

		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		Vec3f Kt = K * hostToNew.translation().cast<float>();

		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for(ImmaturePoint* ph : host->immaturePoints)
		{
			// 利用函数traceOn进行跟踪（又称极线搜索）。
			ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition, 100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
}



//@ 处理挑选出来待激活的点
void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	// 首先生成类 ImmaturePointTemporaryResidual 的对象 tr ，然后利用 optimizeImmaturePoint()进行优化，
	// （注意每次仅对一个点进行优化）优化结果存储在 optimized 。
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
	for(int k=min;k<max;k++)
	{
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}


//@ 激活未成熟点, 加入优化
void FullSystem::activatePointsMT()
{
//[ ***step 1*** ] 阈值计算, 通过距离地图来控制数目
	//currentMinActDist 初值为 2 
	//* 这太牛逼了.....参数
	// （1）根据nPoints确定currentMinActDist，
	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



	FrameHessian* newestHs = frameHessians.back();

	// make dist map.
	// （2）获取最新帧，即当前帧；建立各层金字塔内参；建立dist map。
	coarseDistanceMap->makeK(&Hcalib);
	// 在 makeDistanceMap() 函数里会利用金字塔第一层的内参将除当前帧以外的所有帧的所有点投影到当前帧，将投影位置存储到数组 bfsList1[]。
	// 然后利用 growDistBFS()进行处理，建立 fwdWarpedIDDistFinal ，后续判断能否生成 PointHessian 时用到。（此函数的具体原理还不太清楚）
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

	//coarseTracker->debugPlotDistMap("distMap");

	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000); // 待激活的点

//[ ***step 2*** ] 处理未成熟点, 激活/删除/跳过
	//（3）遍历所有帧的所有 ImmaturePoint ，通过相关参数确定当前遍历点能否生成 PointHessian （即bool canActivate ）。
	// 删除 canActivate 为false的点，对 canActivate 为true的点，将其投影到当前帧上，（这里为什么要用第一层的内参）根据投影位置再进行
	// 判断（使用了上一步生成的 fwdWarpedIDDistFinal ），能够转换为 PointHessian 的点存储到 toOptimize ，否则删除。
	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		if(host == newestHs) continue;

		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
		// 第0层到1层
		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


		for(unsigned int i=0;i<host->immaturePoints.size();i+=1)
		{
			ImmaturePoint* ph = host->immaturePoints[i];
			ph->idxInImmaturePoints = i;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
				//	immature_invalid_deleted++;
				// remove point.
				delete ph;
				host->immaturePoints[i]=0; // 指针赋零
				continue;
			}
			
			//* 未成熟点的激活条件
			// can activate only if this is true.
			// 判断确定当前遍历点能否生成 PointHessian
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
					|| ph->lastTraceStatus == IPS_SKIPPED
					|| ph->lastTraceStatus == IPS_BADCONDITION
					|| ph->lastTraceStatus == IPS_OOB )
							&& ph->lastTracePixelInterval < 8
							&& ph->quality > setting_minTraceQuality
							&& (ph->idepth_max+ph->idepth_min) > 0;


			// if I cannot activate the point, skip it. Maybe also delete it.
			// 删除 canActivate 为false的点，对 canActivate 为true的点，将其投影到当前帧上。  在哪里投影的呢？
			if(!canActivate)
			{
				//* 删除被边缘化帧上的, 和OOB点
				// if point will be out afterwards, delete it instead.
				if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
					// immature_notReady_deleted++;
					delete ph;
					host->immaturePoints[i]=0;
				}
				// immature_notReady_skipped++;
				continue;
			}


			// see if we need to activate point due to distance map.
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			if((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
			{
				// 距离地图 + 小数点
				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

				if(dist>=currentMinActDist* ph->my_type) // 点越多, 距离阈值越大
				{
					coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph);
				}
			}
			else
			{
				delete ph;
				host->immaturePoints[i]=0;
			}
		}
	}


		//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
		//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);
//[ ***step 3*** ] 优化上一步挑出来的未成熟点, 进行逆深度优化, 并得到pointhessian
	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

	// （4）利用activatePointsMT_Reductor()生成PointHessian
	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);

//[ ***step 4*** ] 把PointHessian加入到能量函数, 删除收敛的未成熟点, 或不好的点
	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			//（5）根据上一步优化的结果利用 ef->insertPoint(newpoint) ;和 ef->insertResidual(r) ;将生成的点加入到滑窗优化中。
			ef->insertPoint(newpoint);		// 能量函数中插入点
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);		// 能量函数中插入残差
			assert(newpoint->efPoint != 0);
			// （6）删除未转换成PointHessian的未成熟点。
			delete ph;
		}
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			// bug: 原来的顺序错误
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
			delete ph;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}

//[ ***step 5*** ] 把删除的点丢掉
	for(FrameHessian* host : frameHessians)
	{
		for(int i=0;i<(int)host->immaturePoints.size();i++)
		{
			if(host->immaturePoints[i]==0)
			{
				//bug 如果back的也是空的呢
				host->immaturePoints[i] = host->immaturePoints.back(); // 没有顺序要求, 直接最后一个给空的
				host->immaturePoints.pop_back();
				i--;
			}
		}
	}

// （7）在函数外边重新建立idx：ef->makeIDX();
}






void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

//@ 标记要移除点的状态, 边缘化or丢掉
void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	//if(setting_margPointVisWindow>0)
	{	//bug 又是不用的一条语句
		for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
			if(!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

		for(int i=0; i< (int)frameHessians.size();i++)
			if(frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
	}



	//ef->setAdjointsF();
	//ef->setDeltaF(&Hcalib);
	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		for(unsigned int i=0;i<host->pointHessians.size();i++)
		{
			PointHessian* ph = host->pointHessians[i];
			if(ph==0) continue;

			//* 丢掉相机后面, 没有残差的点
			if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
			{
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				host->pointHessians[i]=0;
				flag_nores++;
			}
			//* 把边缘化的帧上的点, 以及受影响较大的点标记为边缘化or删除
			else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
			{
				flag_oob++;
				//* 如果是一个内点, 则把残差在当前状态线性化, 并计算到零点残差
				if(ph->isInlierNew())
				{
					flag_in++;
					int ngoodRes=0;
					for(PointFrameResidual* r : ph->residuals)
					{
						r->resetOOB();
						r->linearize(&Hcalib);
						r->efResidual->isLinearized = false;
						r->applyRes(true);
						// 如果是激活(可参与优化)的残差, 则给fix住, 计算res_toZeroF
						if(r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
					//* 如果逆深度的协方差很大直接扔掉, 小的边缘化掉
                    if(ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					}
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}


				}
				//* 不是内点直接扔掉
				else
				{
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;


					//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
				}

				host->pointHessians[i]=0; // 把点给删除
			}
		}

		//* 删除边缘化或者删除的点
		for(int i=0;i<(int)host->pointHessians.size();i++)
		{
			if(host->pointHessians[i]==0)
			{
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
	}

}

/********************************
 * @ function:
 * 
 * @ param: 	image		标定后的辐照度和曝光时间
 * @			id			
 * 
 * @ note: start from here
 *******************************/
/*
 * DSO的入口是FullSystem::addActiveFrame，输入的图像生成 FrameHessian 和 FrameShell 的对象， FrameShell 是 FrameHessian 的成员变量，
 * FrameHessian 保存图像信息， FrameShell 保存帧的位置姿态信息。代码中一般用 fh 指针变量指向当前帧的 FrameHessian。在处理完成当前帧之后，
 * 会删除 FrameHessian，而保存 FrameShell 在变量 allFrameHistory 中，作为最后整条轨迹的输出。对输入图像会做预处理，如果有光度标定,像素值
 * 不是灰度值，而是处理后的辐射值，这些辐射值的大小是[0, 255]，float型。数据预处理部分是在FullSystem::addActiveFrame 中调用
 * 的 FrameHessian::makeImages ，这个函数为当前帧的图像建立图像金字塔，并且计算每一层图像的梯度。这些计算结果都存储在 FrameHessian 的成员变量中，
 * 1. dIp 每一层图像的辐射值、x 方向梯度、y 方向梯度；2. dI 指向 dIp[0] 也就是原始图像的信息；3. absSquaredGrad 存储 xy 方向梯度值的平方和。
 */
void FullSystem::addActiveFrame( ImageAndExposure* image, int id )
{
	//[ ***step 1*** ] track线程锁
    if(isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);


	//[ ***step 2*** ] 创建FrameHessian和FrameShell, 并进行相应初始化, 并存储所有帧
	// 初始化存储图像帧信息的类，生成类 FrameHessian ， FrameShell 的对象，并设置初值，
	// 包括： camToWorld ， aff_g2l 等，并将当前帧的信息加入到 allFrameHistory 。
	// =========================== add into allFrameHistory =========================
	// - 如果到了第二帧，首先与第一帧一样，进行图像帧相关信息的初始定义，并设置初值。
	// = 经过八帧之后，此时已经初始化成功，在讨论第九帧的时候，我将其设为非关键帧来介绍对非关键帧的处理。
	// 当前帧生成FrameHessian，FrameShell的对象
	FrameHessian* fh = new FrameHessian(); // 保存图像信息
	FrameShell* shell = new FrameShell(); // 保存帧的位置姿态信息
	// 相关值的初始设置
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	shell->aff_g2l = AffLight(0,0); // 光度仿射变换, 用来建模曝光时间
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;
	fh->shell = shell;
	// 加入到allFrameHistory (所有的历史帧)
	allFrameHistory.push_back(shell);  // 只把简略的shell存起来

	//[ ***step 3*** ] 得到曝光时间, 生成金字塔, 计算整个图像梯度
	// 对当前帧 makeImages()，计算当前图像帧各层金字塔的像素灰度值以及梯度。
	// =========================== make Images / derivatives etc. =========================
	// makeImages()计算梯度，梯度平方和。
	fh->ab_exposure = image->exposure_time; //曝光时间设置
    fh->makeImages(image->image, &Hcalib); // Hcalib 是 CalibHessian 类型



	//[ ***step 4*** ] 进行初始化
	// 前端初始化由 coarseInitializer 类完成。
	// 首先判断是否完成了初始化，如果没有完成初始化，就将当前帧 fh 输入 CoarseInitializer::setFirst 中。完成之后接着处理下一帧。
	// 初始化最少需要有七帧，如果第二帧CoarseInitializer::trackFrame 处理完成之后，位移足够,则再优化到满足位移的后5帧返回true. 
	// 在 FullSystem::initializerFromInitializer 中为第一帧生成 pointHessians，一共2000个左右。随后将第7帧作为 KeyFrame 输入
	// 到 FullSystem::deliverTrackedFrame，最终流入 FullSystem::makeKeyFrame。
	// （FullSystem::deliverTrackedFrame 的作用就是实现多线程的数据输入。）
	if(!initialized)
	{
		// use initializer!
		//[ ***step 4.1*** ] 加入第一帧
		if(coarseInitializer->frameID<0)	// first frame set. fh is kept by coarseInitializer. 所有关键帧的序号，初始值为-1
		{
			// 初始化操作，设置第一帧。
			coarseInitializer->setFirst(&Hcalib, fh);
		}
		// 如果是第二帧，对第一帧进行跟踪
		// 第三，四，五，六，七帧。按照笔者的理解，第3,4,5,6,7帧的处理与第2帧的处理一样，前一帧的优化结果作为后一帧的优化初值。
		// 前面直到else if(coarseInitializer->trackFrame(fh, outputWrapper))的处理均与第2帧一样，在当前帧的时候，
		// 对第一帧进行跟踪之后会返回ture，表示可以进行初始化操作。
		// 前面几帧的处理都是为初始化做准备，一直到第8帧才达到满足进行初始化的条件（如果不出现意外的情况）
		/// DSO 代码中 CoarseInitializer::trackFrame 目的是优化两帧（ref frame 和 new frame）之间的相对状态和 ref frame 中所有点的逆深度。
		else if(coarseInitializer->trackFrame(fh, outputWrapper))	// if SNAPPED
		{
		//[ ***step 4.2*** ] 跟踪成功, 完成初始化
			initializeFromInitializer(fh); //真正的初始化操作
			lock.unlock();
			// 利用deliverTrackedFrame(fh, true)对当前图像帧处理，实现关键帧和非关键帧的区分(当前为true，将当前帧作为关键帧进行处理)
			deliverTrackedFrame(fh, true); // 前端后端的数据接口，进行数据通讯
		}
		else
		{
			// if still initializing
			fh->shell->poseValid = false;
			delete fh;
		}
		return;
	}
	else	// do front-end operation.
	{
//[ ***step 5*** ] 对新来的帧进行跟踪, 得到位姿光度, 判断跟踪状态
		// =========================== SWAP tracking reference?. =========================
		if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			// 交换参考帧和当前帧的coarseTracker
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = coarseTracker; coarseTracker=coarseTracker_forNewKF; coarseTracker_forNewKF=tmp;
		}

		//TODO 使用旋转和位移对像素移动的作用比来判断运动状态
		Vec4 tres = trackNewCoarse(fh); // 对参考帧进行跟踪
		if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            printf("Initial Tracking failed: LOST!\n");
			isLost=true;
            return;
        }
//[ ***step 6*** ] 判断是否插入关键帧
		// 4.确定当前帧是否能作为关键帧，会考虑上一步的跟踪结果，以及当前滑窗内关键帧的数量，判断结果为needToMakeKF。
		bool needToMakeKF = false;
		if(setting_keyframesPerSecond > 0)  // 每隔多久插入关键帧
		{
			needToMakeKF = allFrameHistory.size()== 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
		}
		else
		{
			Vec2 refToFh=AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
					coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			// BRIGHTNESS CHECK
			needToMakeKF = allFrameHistory.size()== 1 ||
					setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +  // 平移像素位移
					setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) + 	//TODO 旋转像素位移, 设置为0???
					setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +	// 旋转+平移像素位移
					setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 ||		// 光度变化大
					2*coarseTracker->firstCoarseRMSE < tres[0];		// 误差能量变化太大(最初的两倍)

		}




        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);



//[ ***step 7*** ] 把该帧发布出去
		lock.unlock();
		// 5.利用deliverTrackedFrame()进行关键帧和非关键帧的区分处理。（在此假设第九帧是非关键帧, needToMakeKF 为false）
		deliverTrackedFrame(fh, needToMakeKF);
		return;
	}
}

//@ 把跟踪的帧, 给到建图线程, 设置成关键帧或非关键帧
// 利用deliverTrackedFrame(fh, true)对当前图像帧处理，实现关键帧和非关键帧的区分(当前为true，将当前帧作为关键帧进行处理)
void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{

	//! 顺序执行
	if(linearizeOperation) 
	{
		if(goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->dI);
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );



		// 当前为true，将当前帧作为关键帧进行处理
		if(needKF) makeKeyFrame(fh);
		// 当前为false，将当前帧作为非关键帧进行处理
		else makeNonKeyFrame(fh);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex); // 跟踪和建图同步锁
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);  // 当没有跟踪的图像, 就一直阻塞trackMapSyncMutex, 直到notify
		}

		lock.unlock();
	}
}

//@ 建图线程
void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0)
		{
			trackedFrameSignal.wait(lock);   // 没有图像等待trackedFrameSignal唤醒
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();


		// guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();		// 运行makeKeyFrame是不会影响unmappedTrackedFrames的, 所以解锁
			makeKeyFrame(fh);
			lock.lock();
			mappedFrameSignal.notify_all();  // 结束前唤醒
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
		{
			lock.unlock();
			makeNonKeyFrame(fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)  // 太多了给处理掉
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
				}
				delete fh;
			}

		}
		else
		{
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)  // 后面需要关键帧
			{
				lock.unlock();
				makeKeyFrame(fh);
				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeNonKeyFrame(fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

//@ 结束建图线程
void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

//@ 设置成非关键帧
void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex); // 生命周期结束后自动解锁
		assert(fh->shell->trackingRef != 0);
		// mapping时将它当前位姿取出来得到camToWorld
		// 1.此处与关键帧的处理一样
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		// 把此时估计的位姿取出来
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	// 2.利用traceNewCoarse(fh);将进行点跟踪，优化关键帧未成熟点的逆深度（因为只对关键帧选点）。
	traceNewCoarse(fh);  // 更新未成熟点(深度未收敛的点)
	delete fh;
}

//@ 生成关键帧, 优化, 激活点, 提取点, 边缘化关键帧
void FullSystem::makeKeyFrame( FrameHessian* fh)
{

//[ ***step 1*** ] 设置当前估计的fh的位姿, 光度参数
	// needs to be set by mapping thread
	{	// 同样取出位姿, 当前的作为最终值
		//? 为啥要从shell来设置 ???   答: 因为shell不删除, 而且参考帧还会被优化, shell是桥梁
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		// 1.获取当前帧的camToWorld，并对当前帧的状态进行setEvalPT_scaled()处理。
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		// 在setEvalPT_scaled()函数里，定义了一个10维向量： initial_state ，并赋值。然后利用函数 setStateScaled()对向量进行处理：
		// 乘以相关参数（SCALE_xxxxx），计算了PRE_worldToCam和PRE_camToWorld。
		// 然后利用setStateZero()函数对上一步计算的state进行处理：计算了一些扰动量以及nullspace。
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l); // 待优化值
	}

//[ ***step 2*** ] 把这一帧来更新之前帧的未成熟点
	// 2.traceNewCoarse(fh);利用当前帧对所有的frameHessians的未成熟点ImmaturePoint进行跟踪，优化其逆深度。注意此时的frameHessians里
	// 只有第一帧，且其生成的未成熟点已全部加入到优化中，因此此时这个函数相当于未运行。（后面再分析）
	traceNewCoarse(fh); // 更新未成熟点(深度未收敛的点)

	boost::unique_lock<boost::mutex> lock(mapMutex); // 建图锁

//[ ***step 3*** ] 选择要边缘化掉的帧
	// =========================== Flag Frames to be Marginalized. =========================
	// 3.判断当前帧是否边缘化并标记：关键帧是否边缘化的判断条件可以从论文中得到，这里不说了，
	// 若当前帧需要边缘化，则fh->flaggedForMarginalization = true; flagged++;
	flagFramesForMarginalization(fh);  // TODO 这里没用最新帧，可以改进下

//[ ***step 4*** ] 加入到关键帧序列
	// =========================== add New Frame to Hessian Struct. =========================
	// 4.将当前帧加入到滑窗优化中，（注意，仅对关键帧执行）
	// 当前帧加入到 vector：frameHessians 和 allKeyFramesHistory ，并利用 insertFrame() 加入到优化中，和 setPrecalcValues() 进行一些预计算。
	// （与对一帧的处理一样，区别在于 frameHessians 里的帧的数量）
	fh->idx = frameHessians.size();
	frameHessians.push_back(fh);
	fh->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);

	setPrecalcValues();	// 每添加一个关键帧都会运行这个来设置位姿, 设置位姿线性化点


//[ ***step 5*** ] 构建之前关键帧与当前帧fh的残差(旧的)
	// 5.利用当前关键帧建立残差项residual.
	// 建立过程为：遍历所有frameHessians，（里面包含当前帧，因此会跳过当前帧）遍历帧的所有pointHessians，建立类PointFrameResidual的对象r，
	// 设置r的状态，并push_back到点的residuals，利用insertResidual添加到滑窗优化中，并设置ph->lastResiduals[]。
	// =========================== add new residuals for old points =========================
	int numFwdResAdde=0;
	for(FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		if(fh1 == fh) continue;
		for(PointHessian* ph : fh1->pointHessians)  // 全都构造之后再删除
		{
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh); // 新建当前帧fh和之前帧之间的残差
			r->setState(ResState::IN);
			ph->residuals.push_back(r);
			ef->insertResidual(r);
			ph->lastResiduals[1] = ph->lastResiduals[0]; // 设置上上个残差
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN); // 当前的设置为上一个
			numFwdResAdde+=1;
		}
	}



//[ ***step 6*** ] 激活所有关键帧上的部分未成熟点(构造新的残差)
	// 6.activatePointsMT();利用当前帧的信息优化所有 frameHessians 的 ImmaturePoint ，注意此时 frameHessians 里面包含了当前帧，因此在遍历的时候会跳过。
	// 同样，此时由于第一帧中没有未成熟点，此函数相当于未执行。
	// =========================== Activate Points (& flag for marginalization). =========================
	activatePointsMT(); // 将未成熟点转化为PointHessian
	// 利用makeIDX()函数重新建立idx。
	// （7） 重新建立idx：ef->makeIDX(); 此处的步骤（7）从 activatePointsMT() 中继续。（此处假设当前的第十帧是关键帧。）
	ef->makeIDX();  // ? 为啥要重新设置ID呢, 是因为加新的帧了么



//[ ***step 7*** ] 对滑窗内的关键帧进行优化(说的轻松, 里面好多问题)
	// 7.后端滑窗优化
	// 首先获取当前帧的frameEnergyTH，然后利用optimize()函数进行优化，优化结果为
	// =========================== OPTIMIZE ALL =========================
	// （8） 滑窗优化（此处假设当前的第十帧是关键帧。）
	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;  // 这两个不是一个值么???
	float rmse = optimize(setting_maxOptIterations);





	// =========================== Figure Out if INITIALIZATION FAILED =========================
	//* 所有的关键帧数小于4，认为还是初始化，此时残差太大认为初始化失败
	// 8.根据优化返回的结果rmse和allKeyFramesHistory.size()判断是否初始化成功。此时仅有2个关键帧，
	// 若rmse大于20*benchmark_initializerSlackFactor，则初始化失败。
	if(allKeyFramesHistory.size() <= 4)
	{
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		// （9）此时总共的关键帧有3个，因此还会根据上一步优化返回的结果判断初始化的质量： （此处假设当前的第十帧是关键帧。）
		// 若rmse > 13*benchmark_initializerSlackFactor 则初始化失败。
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}



    if(isLost) return;  // 优化后的能量函数太大, 认为是跟丢了



//[ ***step 8*** ] 去除外点, 把最新帧设置为参考帧
//TODO 是否可以更加严格一些
	// =========================== REMOVE OUTLIER =========================
	// 9.removeOutliers();对未构成residual的残差点进行处理：利用dropPointsF()函数执行removePoint(p)，然后重新makeIDX();
	// （10） removeOutliers(); （此处假设当前的第十帧是关键帧。）
	removeOutliers();




	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		// （11） （此处假设当前的第十帧是关键帧。）
		coarseTracker_forNewKF->makeK(&Hcalib);  // 更新了内参, 因此重新make
		// 10.setCoarseTrackingRef(frameHessians) 设置当前帧为下次跟踪的参考帧，并通过 makeCoarseDepthL0() 将目标帧是当前帧的点(即构建残差时投影到当前帧的点)
		// 优化的逆深度建立 idepth[0]， weightSums[0]，然后通过对下层采样获取金字塔各层的 idepth_l = idepth[lvl] 和 weightSums_l = weightSums[lvl]。
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);



        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}


	debugPlot("post Optimize");





//[ ***step 9*** ] 标记删除和边缘化的点, 并删除&边缘化
	// =========================== (Activate-)Marginalize Points =========================
	// 11.标记需要边缘化的点并对其进行边缘化操作，前面已经将未构成residual的点进行了删除，此处为了计算的实时性，会对满足一定条件的点进行边缘化处理
	// （具体的不说了，后面会考虑写一篇专门介绍边缘化的博客）。
	// （12） （此处假设当前的第十帧是关键帧。）
	flagPointsForRemoval();
	ef->dropPointsF();  // 扔掉drop的点
	// 每次设置线性化点都会更新零空间
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
	// 边缘化掉点, 加在HM, bM上
	ef->marginalizePointsF();


//[ ***step 10*** ] 生成新的点
	// =========================== add new Immature points & new residuals =========================
	// 12.makeNewTraces(fh, 0);对当前帧利用 pixelSelector->makeMaps()进行选点操作，并且生成 ImmaturePoint ， push_back到 newFrame->immaturePoints。
	// （13） （此处假设当前的第十帧是关键帧。）
	makeNewTraces(fh, 0);





    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }



	// =========================== Marginalize Frames =========================
//[ ***step 11*** ] 边缘化掉关键帧
	//* 边缘化一帧要删除or边缘化上面所有点
	// 13.边缘化关键帧：marginalizeFrame(frameHessians[i]);，边缘化前面标记的关键帧。（同样，具体细节利用边缘化的博客来分析。）
	for(unsigned int i=0;i<frameHessians.size();i++)
		if(frameHessians[i]->flaggedForMarginalization)
			{marginalizeFrame(frameHessians[i]); i=0;}  // （14） （此处假设当前的第十帧是关键帧。）

	// 至此第十帧作为关键帧的操作全部执行完毕


	printLogLine();
    //printEigenValLine();

}

//@ 从初始化中提取出信息, 用于跟踪.
// FullSystem::initializeFromInitializer，第一帧是 firstFrame，第七帧是 newFrame，从 CoarseInitializer 中抽取出 2000 个点
// 作为 firstFrame 的 pointHessians。设置的逆深度有 CoarseIntiailzier::trackFrame 中计算出来的 iR 和 idepth，而这里
// 使用了 rescaleFactor 这个局部变量，保证所有 iR 的均值为 1。iR 设置的是 PointHessian 的 idepth，
// 而 idepth 设置的是 PointHessian 的 idepth_zero(缩放了scale倍的固定线性化点逆深度)，idepth_zero 相当于估计的真值，用于计算误差。
void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	//[ ***step 1*** ] 把第一帧设置成关键帧, 加入队列, 加入EnergyFunctional
	// add firstframe. 添加第一帧
	// 在这里新定义vector：frameHessians和allKeyFramesHistory，并把第一帧加入进去，
	// （这两个容器应该是只存储关键帧的相关信息，此时里面仅有第一帧）
	FrameHessian* firstFrame = coarseInitializer->firstFrame;  // 第一帧增加进地图
	firstFrame->idx = frameHessians.size(); // 赋值给它id (0开始)
	frameHessians.push_back(firstFrame);  	// 地图内关键帧容器
	firstFrame->frameID = allKeyFramesHistory.size();  	// 所有历史关键帧id
	allKeyFramesHistory.push_back(firstFrame->shell); 	// 所有历史关键帧
	// 第一帧加入优化：
	// 利用ef->insertFrame(firstFrame, &Hcalib)，将第一帧加入到优化后端energyFunction
	// Hcalib 是相机响应函数
	ef->insertFrame(firstFrame, &Hcalib);
	//在setPrecalcValues();会建立所有帧的目标帧，并且进行主导帧和目标帧之间相对状态的预计算，实现的函数见函数.set()，
	// 计算的量有：leftToLeft，PRE_RTll，PRE_tTll，PRE_KRKiTll，PRE_RKiTll，PRE_KtTll，PRE_aff_mode，PRE_b0_mode。
	setPrecalcValues();   		// 设置相对位姿预计算值

	//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

	// 初始第一帧的各种点，包括：pointHessians，pointHessiansMarginalized，pointHessiansOut
	firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f); // 20%的点数目
	firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f); // 被边缘化
	firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f); // 丢掉的点

	//[ ***step 2*** ] 求出平均尺度因子
	// 利用前面帧对第一帧的跟踪更新的逆深度，计算尺度因子 rescaleFactor(相对的)，并且根据相关参数设置第一帧保持的点的数目：keepPercentage
	float sumID=1e-5, numID=1e-5;
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		//? iR的值到底是啥
		sumID += coarseInitializer->points[0][i].iR; // 第0层点的中位值, 相当于
		numID++;
	}
	float rescaleFactor = 1 / (sumID / numID);  // 求出尺度因子

	// randomly sub-select the points I need.
	// 目标点数 / 实际提取点数
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if(!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
                (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

	//[ ***step 3*** ] 创建PointHessian, 点加入关键帧, 加入EnergyFunctional
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		if(rand()/(float)RAND_MAX > keepPercentage) continue; // 如果提取的点比较少, 不执行; 提取的多, 则随机干掉

		// 将第一帧的未成熟点生成PointHessian，并且设置PointHessian的相关参数，存储到第一帧的容器pointHessians中，然后
		// 利用insertPoint()加入到后端优化中。注意：前面分析对第一帧选点时说到是在每层图像金字塔都会进行选点，但是此时
		// 生成PointHessian仅利用第0层（即原始图像层）选取的点。
		// 在应用构造函数创建类ImmaturePoint的对象pt时，会计算点的权重：weights[idx]。
		Pnt* point = coarseInitializer->points[0]+i;
		ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);

		if(!std::isfinite(pt->energyTH)) { delete pt; continue; }  // 点值无穷大

		// 创建ImmaturePoint就为了创建PointHessian? 是为了接口统一吧
		pt->idepth_max=pt->idepth_min=1;
		PointHessian* ph = new PointHessian(pt, &Hcalib);
		delete pt;
		if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

		ph->setIdepthScaled(point->iR*rescaleFactor);  //? 为啥设置的是scaled之后的
		ph->setIdepthZero(ph->idepth);			//! 设置初始先验值, 还有神奇的求零空间方法
		ph->hasDepthPrior=true;
		ph->setPointStatus(PointHessian::ACTIVE);	// 激活点

		firstFrame->pointHessians.push_back(ph);
		// 在insertPoint()中会生成PointHessian类型的点ph的EFPoint类型的efp，efp，包含点ph以及其主导帧host。
		// 按照push_back的先后顺序对idxInPoints进行编号， nPoints表示后端优化中点的数量。
		ef->insertPoint(ph);
	}


	//[ ***step 4*** ] 设置第一帧和最新帧的待优化量, 参考帧
	// 通过前面所有帧对第一帧的track以及optimization得到第一帧到第八帧的位姿： firstToNew ，并对平移部分利用尺度因子进行处理。
	SE3 firstToNew = coarseInitializer->thisToNext;
	firstToNew.translation() /= rescaleFactor;


	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		// 设置第一帧和第八帧的相关参数
		firstFrame->shell->camToWorld = SE3();		// 空的初值?
		firstFrame->shell->aff_g2l = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();

		newFrame->shell->camToWorld = firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();

	}

	// 初始化成功：initialized=true;至此初始化已经成功了，接下的操作是将第八帧作为关键帧进行处理。
	initialized=true;
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

//@ 提取新的像素点用来跟踪
// 12.makeNewTraces(fh, 0);对当前帧利用 pixelSelector->makeMaps()进行选点操作，并且生成 ImmaturePoint ， push_back到 newFrame->immaturePoints
void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
	pixelSelector->allowFast = true;  //bug 没卵用
	//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

	newFrame->pointHessians.reserve(numPointsTotal*1.2f);
	//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);


	for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
	for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
	{
		int i = x+y*wG[0];
		if(selectionMap[i]==0) continue;

		ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);
		if(!std::isfinite(impt->energyTH)) delete impt;  // 投影得到的不是有穷数
		else newFrame->immaturePoints.push_back(impt);

	}
	//printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}

//* 计算frameHessian的预计算值, 和状态的delta值
//@ 设置关键帧之间的关系
// 在setPrecalcValues();会建立所有帧的目标帧，并且进行主导帧和目标帧之间相对状态的预计算，实现的函数见函数.set()，
// 计算的量有：leftToLeft，PRE_RTll，PRE_tTll，PRE_KRKiTll，PRE_RKiTll，PRE_KtTll，PRE_aff_mode，PRE_b0_mode。
void FullSystem::setPrecalcValues()
{
	for(FrameHessian* fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size()); // 每个目标帧预运算容器, 大小是关键帧数
		for(unsigned int i=0;i<frameHessians.size();i++)  //? 还有自己和自己的???
			// 计算的量有：leftToLeft，PRE_RTll，PRE_tTll，PRE_KRKiTll，PRE_RKiTll，PRE_KtTll，PRE_aff_mode，PRE_b0_mode。
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib); // 计算Host 与 target之间的变换关系
	}

	// 然后利用：ef->setDeltaF(&Hcalib);建立相关量的微小扰动，包括：adHTdeltaF[idx]，f->delta，f->delta_prior。
	ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
	if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l.a,
                allKeyFramesHistory.back()->aff_g2l.b,
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				ef->resInA << " "  <<
				ef->resInL << " "  <<
				ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				frameHessians.back()->aff_g2l().a << " "  <<
				frameHessians.back()->aff_g2l().b << " "  <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
				(int)frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}


}



void FullSystem::printEigenValLine()
{
	if(!setting_logStuff) return;
	if(ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}





}
