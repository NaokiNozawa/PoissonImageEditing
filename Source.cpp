#include <Eigen/Sparse>
#include <Eigen/Eigen>//�܂��͂�����
#include <iostream>
#include <includeopencv.h>
#include <vector>


void output(std::string str2,std::vector<std::vector<int>> array);


int main()
{

	cv::Mat src_img = cv::imread("source.jpg",1); //�\�[�X�摜�s��
	cv::Mat dtn_img = cv::imread("destination.jpg",1);//�f�X�e�B�l�[�V�����摜�s��
	cv::Mat msk_img = cv::imread("mask.jpg",0);//�}�X�N�摜�s��

	cv::Mat tgt_img = cv::Mat::zeros(msk_img.rows,msk_img.cols,CV_8UC3);

	if((!src_img.data)||(!dtn_img.data)||(!msk_img.data)){
		std::cout << "�摜������܂���" << std::endl;
	}

	int i=0;//number�ɑΉ��t�������邽�߂̔ԍ��p��
	//int t[2] = {-74,78};//���s�ړ��ɑΉ�����x�N�g��

	std::vector<std::vector<int>> array;//�z��錾

	//cv::Mat number = cv::Mat::ones(msk_img.rows,msk_img.cols,CV_8S)*(-1);//�͈͂Ɍ��E������������_��
	Eigen::MatrixXd number = Eigen::MatrixXd::Ones(msk_img.rows,msk_img.cols)*(-1);//�ԍ��Ή��̂��߂̔z��p��

	//std::cout << number << std::endl;

	for(int y=0; y<msk_img.rows; y++){//y�ǂݍ��݊J�n

		for(int x=0; x<msk_img.cols; x++){//x�ǂݍ��݊J�n
			if(msk_img.at<uchar>(y,x)>250){//�����}�X�N�̂��̈ʒu�����Ȃ�

				std::vector<int> v;//�x�N�g���錾
				v.push_back(y);//y�����ɑ��
				v.push_back(x);//����v�ɑ��

				number.coeffRef(y,x) = i;//���W���͂Ŕԍ��������� (y_i, x_i)��i�Ԗ�

				array.push_back(v);//array[i][0]=y_i,
				//array[i][1]=x_i,

				i++;//i�̒l��1�ӂ₷
			}

		}

	}

	//std::cout << number << std::endl;

	std::cout << "�}�X�N�ɑΉ������f�̐��ii�j=" << i << std::endl;


	Eigen::MatrixXd G = Eigen::MatrixXd::Zero(i,3);//���͂ɑΉ�����x�N�g��

	Eigen::SparseMatrix<int> A(i, i);
	A.setZero();

	//std::cout << A <<std::endl;


	for(int s = 0; s < i; s++){
		//�Ίp����
		A.coeffRef(s,s)=-4;//Ref�t���Ȃ��ƕύX�ł��Ȃ�

		for(int c=0; c<3; c++){
			//���͂ɑΉ�����x�N�g��
			G.coeffRef(s,c) = -4 * src_img.at<cv::Vec3b>(array[s][0],array[s][1])[c] +//���͑Ή��̋P�x�l�ω��iAF=G��G�j
				src_img.at<cv::Vec3b>(array[s][0] -1 ,array[s][1])[c] + src_img.at<cv::Vec3b>(array[s][0] + 1,array[s][1])[c] +
				src_img.at<cv::Vec3b>(array[s][0],array[s][1] - 1)[c] + src_img.at<cv::Vec3b>(array[s][0],array[s][1] + 1)[c];
			//std::cout << G.coeffRef(s,c) << std::endl; //�m�F�p



			//���ډ�f�̈��̉�f
			if(number.coeffRef(array[s][0]-1,array[s][1]) == -1){//s�ɑΉ������f�̈�オ�}�X�N�O��������
				G.coeffRef(s,c) -= dtn_img.at<cv::Vec3b>(array[s][0]-1+150,array[s][1]+130)[c];//�^�[�Q�b�g�̋��E����Q��
				//std::cout << G(s) << std::endl; //�m�F�p
			}



			//���ډ�f�̈���̉�f
			if(number.coeffRef(array[s][0]+1,array[s][1]) == -1){//s�Ή��̉�f�̈�����}�X�N�O��������
				G.coeffRef(s,c) -= dtn_img.at<cv::Vec3b>(array[s][0]+1+150,array[s][1]+130)[c];//�^�[�Q�b�g�̋��E����Q��
				//std::cout << G(s) << std::endl;//�m�F�p
			}



			//���ډ�f�̈���̉�f
			if(number.coeffRef(array[s][0],array[s][1]-1) == -1){//s�Ή��̉�f�̈�����}�X�N�O��������
				G.coeffRef(s,c) -= dtn_img.at<cv::Vec3b>(array[s][0]+150,array[s][1]-1+130)[c];//�^�[�Q�b�g�̋��E����Q��
			}


			//���ډ�f�̈�E�̉�f
			if(number.coeffRef(array[s][0],array[s][1]+1) == -1){//s�Ή��̉�f�̈�E���}�X�N�O��������
				G.coeffRef(s,c) -= dtn_img.at<cv::Vec3b>(array[s][0]+150,array[s][1]+1+130)[c];//�^�[�Q�b�g�̋��E����Q��
			}
		}



		if(number.coeffRef(array[s][0]-1,array[s][1]) != -1){
			int s_up = number.coeffRef(array[s][0]-1,array[s][1]);//A���̑Ή�������1���� ��
			A.coeffRef(s,s_up)=1;
			//std::cout << s_up << " " << s << std::endl;
		}

		if(number.coeffRef(array[s][0]+1,array[s][1]) != -1){
			int s_down = number.coeffRef(array[s][0]+1,array[s][1]);//A���̑Ή�������1�����@��
			//std::cout << s_down << " " << s << std::endl;
			A.coeffRef(s,s_down)=1;
		}

		if(number.coeffRef(array[s][0],array[s][1]-1) != -1){//A���̑Ή�������1�����@��
			A.coeffRef(s,s-1)=1;
		}

		if(number.coeffRef(array[s][0],array[s][1]+1) != -1){//A���̑Ή�������1�����@�E
			A.coeffRef(s,s+1)=1;
		}
	}
	//std::cout << "no." << s << std::endl;
	std::cout << "loop end" << std::endl;
	//std::cout << G << std::endl;



	//std::cout << A << std::endl;

	//F = A.inverse() * G; //�t�s��ł��s����

	//Eigen::FullPivLU < Eigen::MatrixXd > lu(A);//LU����


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