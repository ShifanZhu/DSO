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
#include "FullSystem/ImmaturePoint.h"
#include "math.h"

namespace dso
{


//@ 优化未成熟点逆深度, 并创建成PointHessian
// 在 optimizeImmaturePoint() 函数中，将除了该点的主导帧之外的所有关键帧作为该点的目标帧，初始 ImmaturePointTemporaryResidual 类型的 residual ；
// 利用 linearizeResidual()函数进行残差，优化矩阵， energy 等信息的计算；利用上步计算的优化矩阵求解迭代增量，重新计算 energy 和相关优化矩阵，误差减小
// 则接受优化，进行迭代，误差增大则调整 lambda 重新迭代。迭代优化结束之后，根据优化结果生成 PointHessian* p，存储到 vector：:optimized 并建立
// 残差项： PointFrameResidual* r，其中包括相关参数的设置。
PointHessian* FullSystem::optimizeImmaturePoint(
		ImmaturePoint* point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
{
//[ ***step 1*** ] 初始化和其它关键帧的res(点在其它关键帧上投影)
	int nres = 0;
	for(FrameHessian* fh : frameHessians)
	{
		if(fh != point->host)  // 没有创建和自己的
		{
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh;
			nres++; // 观测数
		}
	}
	assert(nres == ((int)frameHessians.size())-1);

	bool print = false;//rand()%50==0;

	float lastEnergy = 0;
	float lastHdd=0;
	float lastbd=0;
	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f;





//[ ***step 2*** ] 使用类LM(GN)的方法来优化逆深度, 而不是使用三角化
//TODO 这种优化求的方法, 和三角化的方法, 哪个更好些
	for(int i=0;i<nres;i++)
	{
		// 利用 linearizeResidual()函数进行残差，优化矩阵， energy 等信息的计算
		lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i,lastHdd, lastbd, currentIdepth);
		residuals[i].state_state = residuals[i].state_NewState;
		residuals[i].state_energy = residuals[i].state_NewEnergy;
	}

	if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
	{
		if(print)
			printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
				nres, lastHdd, lastEnergy);
		return 0;
	}

	if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
			nres, lastHdd,lastEnergy,currentIdepth);

	// 利用上步计算的优化矩阵求解迭代增量，重新计算energy和相关优化矩阵，误差减小则接受优化，进行迭代，误差增大则调整lambda重新迭代
	float lambda = 0.1;
	for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
	{
		float H = lastHdd;
		H *= 1+lambda;
		float step = (1.0/H) * lastbd;
		float newIdepth = currentIdepth - step;

		float newHdd=0; float newbd=0; float newEnergy=0;
		for(int i=0;i<nres;i++)
			newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i,newHdd, newbd, newIdepth);

		if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
		{
			if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres,
					newHdd,
					lastEnergy);
			return 0;
		}

		if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				"",
				lastEnergy, newEnergy, newIdepth);

		if(newEnergy < lastEnergy)
		{
			currentIdepth = newIdepth;
			lastHdd = newHdd;
			lastbd = newbd;
			lastEnergy = newEnergy;
			for(int i=0;i<nres;i++)
			{
				residuals[i].state_state = residuals[i].state_NewState;
				residuals[i].state_energy = residuals[i].state_NewEnergy;
			}

			lambda *= 0.5;
		}
		else
		{
			lambda *= 5;
		}

		if(fabsf(step) < 0.0001*currentIdepth)
			break;
	}

	if(!std::isfinite(currentIdepth))
	{
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
		// 丢弃无穷的点
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

	//* 所有观测里面统计good数, 小了则返回
	int numGoodRes=0;
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) numGoodRes++;

	if(numGoodRes < minObs)
	{
		if(print) printf("OptPoint: OUTLIER!\n");
		//! niubility
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}


//[ ***step 3*** ] 把可以的点创建成PointHessian
	// 迭代优化结束之后，根据优化结果生成 PointHessian* p ，存储到 optimized 并建立残差项：PointFrameResidual* r，其中包括相关参数的设置。
	PointHessian* p = new PointHessian(point, &Hcalib);
	if(!std::isfinite(p->energyTH)) {delete p; return (PointHessian*)((long)(-1));} // 丢弃无穷的点

	p->lastResiduals[0].first = 0;
	p->lastResiduals[0].second = ResState::OOB;
	p->lastResiduals[1].first = 0;
	p->lastResiduals[1].second = ResState::OOB;
	p->setIdepthZero(currentIdepth); // 设置线性化处逆深度
	p->setIdepth(currentIdepth);
	p->setPointStatus(PointHessian::ACTIVE);

	//* 计算PointFrameResidual
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN)
		{
			PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
			r->state_NewEnergy = r->state_energy = 0;
			r->state_NewState = ResState::OUTLIER;
			r->setState(ResState::IN);
			p->residuals.push_back(r);

			if(r->target == frameHessians.back()) // 和最新帧的残差
			{
				p->lastResiduals[0].first = r;
				p->lastResiduals[0].second = ResState::IN;
			}
			else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2])) // 和最新帧上一帧
			{
				p->lastResiduals[1].first = r;
				p->lastResiduals[1].second = ResState::IN;
			}
		}

	if(print) printf("point activated!\n");

	statistics_numActivatedPoints++;
	return p;
}



}
