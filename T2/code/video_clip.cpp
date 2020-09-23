#include<iostream>
#include<opencv2/opencv.hpp>
#define START_SECOND 26
#define SKIP_TO_MIN 61
#define COMPLEMENT_SECOND 9
using namespace std;
using namespace cv;
string timelen2(string a) {
	return a.size() == 2 ? a : '0' + a;
}
string time_stamp(int i,int k ) {
	vector<string> s = { "00","15","30","45" };
	int min = i > 60 ? i % 60 : i;
	int hour = i / 60;
	return timelen2(to_string(hour)) + "_" + timelen2(to_string(min)) + "_"+s[k] ;
}
int main() {
	/*video clip*/
	VideoCapture V("C:/Users/Administrator/Desktop/math/2020��E��/������Ƶ2/Fog20200313000026.mp4");
	Mat frame;
	int i = 0;//FARMES
	int j = 1;//MINS
	int  rate = V.get(cv::CAP_PROP_FPS);
	long startframe = (60 - START_SECOND) * rate;
	for (int k = 0; k < startframe; k++) {
		V >> frame;
	} //��ʼʱ��Ϊ00:01:00
	int k = 1; //seconds*15
	while (!frame.empty()) {
		if (i == 15 * rate) {  //��֡�ɼ�
			if (j == 47 && k == 3) {//48֡�쳣֡�ų�
				j = SKIP_TO_MIN;
				int d = COMPLEMENT_SECOND  * rate; //�����Ƕ���ʱ��� ��Ϊ���쳣ʱ�̲ɼ�ͼ��ʱ��Ϊ1:1:36 ���뵽1:1:45 
				while ( d ) {
					d--;
					V >> frame;
				}
			}
			if (k==4) {
				j += 1;
				k = 0;
			}
			i = 0;
			string time = time_stamp(j,k);//��¼ʱ��
			imwrite("C:/Users/Administrator/Desktop/math/data/clip/"+time+".png", frame);
			k++;
		}
		i++;
		V >> frame;
	}
	return 0;
}