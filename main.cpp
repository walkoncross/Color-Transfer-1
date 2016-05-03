#include<iostream>
#include<opencv2/opencv.hpp>
#include<Eigen/Core>
#include<opencv2/core/core.hpp>
#include<opencv2/core/eigen.hpp>

int main(int argc, char* argv[]) {
	/////////////////////////
	///　必要な変数の宣言 ///
	/////////////////////////


    // Transformation from RGB to LMS
	cv::Mat1f RGB2LMS = (cv::Mat_<float>(3, 3) << 0.3811, 0.5783, 0.0402, 0.1967, 0.7244, 0.0782, 0.0241, 0.1288, 0.8444);

	// Transformation from LMS to RGB
	cv::Mat1f LMS2RGB = (cv::Mat_<float>(3, 3) << 4.4679, -3.5873, 0.1193, -1.2186, 2.3809, -0.1624, 0.0497, -0.2439, 1.2045);

	// First transformation from LMS to lab
	cv::Mat1f LMS2lab1 = (cv::Mat_<float>(3, 3) << 1.0 / sqrt(3.0), 0.0, 0.0, 0.0, 1.0 / sqrt(6.0), 0.0, 0.0, 0.0, 1.0 / sqrt(2.0));

	// Second transformation from LMS to lab
	cv::Mat1f LMS2lab2 = (cv::Mat_<float>(3, 3) << 1.0, 1.0, 1.0, 1.0, 1.0, -2.0, 1.0, -1.0, 0.0);

	// Transformation from LMS to lab
	cv::Mat1f LMS2lab;
	LMS2lab = LMS2lab1 * LMS2lab2;

	// Transformation from lab to LMS
	cv::Mat1f lab2LMS = LMS2lab.inv();

	
	const float eps = 1.0e-4;

	//平均値を格納する変数(各チャンネルごとに用意)
	cv::Mat1f src_mean(1, 3);
	cv::Mat1f ref_mean(1, 3);
	//分散を格納する変数
	cv::Mat1f src_disp(1, 3);
	cv::Mat1f ref_disp(1, 3);
	//標準偏差を格納する変数
	cv::Mat1f src_st(1, 3);
	cv::Mat1f ref_st(1, 3);
	for (int c = 0; c < 3; ++c) {
		src_mean(c) = 0.0;
		src_disp(c) = 0.0;
		ref_mean(c) = 0.0;
		ref_disp(c) = 0.0;
	}

	////////////////////////
	/// 以下、プログラム ///
	////////////////////////

	//画像の読み込み
	cv::Mat src = cv::imread(argv[1]);
	cv::Mat ref = cv::imread(argv[2]);
	cv::namedWindow("source", cv::WINDOW_AUTOSIZE|cv::WINDOW_FREERATIO);
	cv::imshow("source", src);
	cv::namedWindow("reference", cv::WINDOW_AUTOSIZE|cv::WINDOW_FREERATIO);
	cv::imshow("reference", ref);

	//BGRからRGBに変換
	cv::cvtColor(src, src, CV_BGR2RGB);
	cv::cvtColor(ref, ref, CV_BGR2RGB);

	//0~1のfloatに変換
	src.convertTo(src, CV_64FC3, 1.0 / 255.0);
	ref.convertTo(ref, CV_64FC3, 1.0 / 255.0);

	cv::Mat3f src3f = cv::Mat3f(src);
	cv::Mat3f ref3f = cv::Mat3f(ref);

	cv::Mat1f buf(1, 3);

	// srcの色空間をRGBからlabに変換
	for (int y = 0; y < src.rows; ++y) {
		for (int x = 0; x < src.cols; ++x) {
			// RGB -> LMS
			buf = RGB2LMS * cv::Mat1f(src3f(y, x));

			// log10を取る（小さいの処理も含む）
			for (int c = 0; c < 3; ++c) {
				buf(c) = buf(c) > eps ? log10(buf(c)) : log10(eps);
			}

			// LMS -> lab
			buf = LMS2lab * buf;

			src3f(y, x) = buf;

			for (int c = 0; c < 3; ++c) {
				src_mean(c) += buf(c);
				src_disp(c) += buf(c) * buf(c);
			}
		}
	}

	// refの色空間をRGBからlabに変換
	for (int y = 0; y < ref.rows; ++y) {
		for (int x = 0; x < ref.cols; ++x) {
			// RGB -> LMS
			buf = RGB2LMS * cv::Mat1f(ref3f(y, x));

			// log10を取る（小さいの処理も含む）
			for (int c = 0; c < 3; ++c) {
				buf(c) = buf(c) > eps ? log10(buf(c)): log10(eps);
			}

			// LMS -> lab
			buf = LMS2lab * buf;

			ref3f(y, x) = buf;

			for (int c = 0; c < 3; ++c) {
				ref_mean(c) += buf(c);
				ref_disp(c) += buf(c) * buf(c);
			}
		}
	}

	//平均と分散の計算
	for (int c = 0; c < 3; ++c) {
		src_mean(c) /= src.rows * src.cols;
		src_disp(c) = (src_disp(c) / (src.rows * src.cols)) - src_mean(c) * src_mean(c);
		src_st(c) = sqrt(src_disp(c));
		ref_mean(c) /= ref.rows * ref.cols;
		ref_disp(c) = (ref_disp(c) / (ref.rows * ref.cols)) - ref_mean(c) * ref_mean(c);
		ref_st(c) = sqrt(ref_disp(c));
	}

	std::cout << src_mean << std::endl;
	std::cout << src_disp << std::endl;
	std::cout << ref_mean << std::endl;
	std::cout << ref_disp << std::endl;

	// Color Transfer
	for (int y = 0; y < src.rows; ++y) {
		for (int x = 0; x < src.cols; ++x) {
			for (int c = 0; c < 3; ++c) {
				src3f(y, x)(c) = (src3f(y, x)(c) - src_mean(c)) * ref_st(c) / src_st(c) + ref_mean(c);
			}
		}
	}

	//lab -> RGB
	for (int y = 0; y < src.rows; ++y) {
		for (int x = 0; x < src.cols; ++x) {
			//lab -> LMS
			buf = lab2LMS * cv::Mat1f(src3f(y, x));

			for (int c = 0; c < 3; ++c) {
				buf(c) = buf(c) > -4.0 ? pow(10.0, buf(c)) : eps;
			}

			buf = LMS2RGB * buf;

			src3f(y, x) = buf;
		}
	}

	// src3f を src に戻す
	src = cv::Mat(src3f);

	// uchar に戻す
	src.convertTo(src, CV_8UC3, 255.0);

	// RGB -> BGR
	cv::cvtColor(src, src, CV_RGB2BGR);

	// 結果の表示
	cv::namedWindow("result");
	cv::imshow("result", src);
	cv::imwrite("result.jpg", src);

	cv::waitKey(0);
	cv::destroyAllWindows();
	system("pause");
	return 0;
}
