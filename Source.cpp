#include <Eigen/Sparse>
#include <Eigen/Eigen>//まずはこっち
#include <iostream>
#include <includeopencv.h>
#include <vector>


void output(std::string str2,std::vector<std::vector<int>> array);


int main()
{

	cv::Mat src_img = cv::imread("source.jpg",1); //ソース画像行列化
	cv::Mat dtn_img = cv::imread("destination.jpg",1);//デスティネーション画像行列化
	cv::Mat msk_img = cv::imread("mask.jpg",0);//マスク画像行列化

	cv::Mat tgt_img = cv::Mat::zeros(msk_img.rows,msk_img.cols,CV_8UC3);

	if((!src_img.data)||(!dtn_img.data)||(!msk_img.data)){
		std::cout << "画像がありません" << std::endl;
	}

	int i=0;//numberに対応付けさせるための番号用意
	//int t[2] = {-74,78};//平行移動に対応するベクトル

	std::vector<std::vector<int>> array;//配列宣言

	//cv::Mat number = cv::Mat::ones(msk_img.rows,msk_img.cols,CV_8S)*(-1);//範囲に限界があったからダメ
	Eigen::MatrixXd number = Eigen::MatrixXd::Ones(msk_img.rows,msk_img.cols)*(-1);//番号対応のための配列用意

	//std::cout << number << std::endl;

	for(int y=0; y<msk_img.rows; y++){//y読み込み開始

		for(int x=0; x<msk_img.cols; x++){//x読み込み開始
			if(msk_img.at<uchar>(y,x)>250){//もしマスクのその位置が白なら

				std::vector<int> v;//ベクトル宣言
				v.push_back(y);//yをｖに代入
				v.push_back(x);//ｘをvに代入

				number.coeffRef(y,x) = i;//座標入力で番号が得られる (y_i, x_i)はi番目

				array.push_back(v);//array[i][0]=y_i,
				//array[i][1]=x_i,

				i++;//iの値を1ふやす
			}

		}

	}

	//std::cout << number << std::endl;

	std::cout << "マスクに対応する画素の数（i）=" << i << std::endl;


	Eigen::MatrixXd G = Eigen::MatrixXd::Zero(i,3);//入力に対応するベクトル

	Eigen::SparseMatrix<int> A(i, i);
	A.setZero();

	//std::cout << A <<std::endl;


	for(int s = 0; s < i; s++){
		//対角成分
		A.coeffRef(s,s)=-4;//Ref付けないと変更できない

		for(int c=0; c<3; c++){
			//入力に対応するベクトル
			G.coeffRef(s,c) = -4 * src_img.at<cv::Vec3b>(array[s][0],array[s][1])[c] +//入力対応の輝度値変化（AF=GのG）
				src_img.at<cv::Vec3b>(array[s][0] -1 ,array[s][1])[c] + src_img.at<cv::Vec3b>(array[s][0] + 1,array[s][1])[c] +
				src_img.at<cv::Vec3b>(array[s][0],array[s][1] - 1)[c] + src_img.at<cv::Vec3b>(array[s][0],array[s][1] + 1)[c];
			//std::cout << G.coeffRef(s,c) << std::endl; //確認用



			//注目画素の一個上の画素
			if(number.coeffRef(array[s][0]-1,array[s][1]) == -1){//sに対応する画素の一個上がマスク外だったら
				G.coeffRef(s,c) -= dtn_img.at<cv::Vec3b>(array[s][0]-1+150,array[s][1]+130)[c];//ターゲットの境界上を参照
				//std::cout << G(s) << std::endl; //確認用
			}



			//注目画素の一個下の画素
			if(number.coeffRef(array[s][0]+1,array[s][1]) == -1){//s対応の画素の一個下がマスク外だったら
				G.coeffRef(s,c) -= dtn_img.at<cv::Vec3b>(array[s][0]+1+150,array[s][1]+130)[c];//ターゲットの境界上を参照
				//std::cout << G(s) << std::endl;//確認用
			}



			//注目画素の一個左の画素
			if(number.coeffRef(array[s][0],array[s][1]-1) == -1){//s対応の画素の一個左がマスク外だったら
				G.coeffRef(s,c) -= dtn_img.at<cv::Vec3b>(array[s][0]+150,array[s][1]-1+130)[c];//ターゲットの境界上を参照
			}


			//注目画素の一個右の画素
			if(number.coeffRef(array[s][0],array[s][1]+1) == -1){//s対応の画素の一個右がマスク外だったら
				G.coeffRef(s,c) -= dtn_img.at<cv::Vec3b>(array[s][0]+150,array[s][1]+1+130)[c];//ターゲットの境界上を参照
			}
		}



		if(number.coeffRef(array[s][0]-1,array[s][1]) != -1){
			int s_up = number.coeffRef(array[s][0]-1,array[s][1]);//A内の対応部分に1を代入 上
			A.coeffRef(s,s_up)=1;
			//std::cout << s_up << " " << s << std::endl;
		}

		if(number.coeffRef(array[s][0]+1,array[s][1]) != -1){
			int s_down = number.coeffRef(array[s][0]+1,array[s][1]);//A内の対応部分に1を代入　下
			//std::cout << s_down << " " << s << std::endl;
			A.coeffRef(s,s_down)=1;
		}

		if(number.coeffRef(array[s][0],array[s][1]-1) != -1){//A内の対応部分に1を代入　左
			A.coeffRef(s,s-1)=1;
		}

		if(number.coeffRef(array[s][0],array[s][1]+1) != -1){//A内の対応部分に1を代入　右
			A.coeffRef(s,s+1)=1;
		}
	}
	//std::cout << "no." << s << std::endl;
	std::cout << "loop end" << std::endl;
	//std::cout << G << std::endl;



	//std::cout << A << std::endl;

	//F = A.inverse() * G; //逆行列でも行ける

	//Eigen::FullPivLU < Eigen::MatrixXd > lu(A);//LU分解


	Eigen::MatrixXd F(i,3);
	for(int c=0; c<3; c++){

		//Eigen::VectorXd d = G.col(c);
		//F.col(c)=lu.solve(d);

		//F.col(c) = A.inverse() * G.col(c);

		//Eigen::SimplicialCholesky<Eigen::SparseMatrix<int> > solver(A);
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double> > solver(A.cast<double>());
		F.col(c) = solver.solve(G.col(c));
		//F = solver.solve(G);


		for(int s=0; s<i; s++){
			if(F(s,c) > 255){
				F(s,c) = 255;
			}
			if(F(s,c)<0){
				F(s,c)=0;
			}
		}

	}
	//std::cout<< "F=" << F << std::endl;

	//std::cout << "AF-G=" << A*F-G << std::endl;

	tgt_img = dtn_img;

	for(int s=0; s<i; s++){
		for(int c=0; c<3; c++){
			tgt_img = dtn_img;
			tgt_img.at<cv::Vec3b>(array[s][0]+150,array[s][1]+130)[c] = F(s,c);
		}
	}




	cv::imshow("tgt_img.jpeg", tgt_img);
	//std::cout << tgt_img << std::endl;

	cv::Mat disp = cv::Mat::zeros(cv::Size(320, 240), CV_8UC1);
	cv::imshow("dummy", disp);
	int key = cv::waitKey(0);

	return 0;
}