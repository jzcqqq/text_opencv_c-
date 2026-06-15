#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <string>
#define ID_SHOW_TME 30;
class Defect {
public:
	Defect(int state, int x, int y, int w, int h)
		: state(state), x(x), y(y), w(w), h(h) {
	}

	int getState() const { return state; }
	int getX() const { return x; }
	int getY() const { return y; }
	int getW() const { return w; }
	int getH() const { return h; }

private:
	int state;
	int x, y, w, h;
};

class Product {
public:

	static int scratch_sum;
	static int blot_sum;
	static int count;
	bool detect_finish;
	int showframe;

	Product(int id, int cx, int cy, int x, int y, int w, int h)
		: pid(id), centre_x(cx), centre_y(cy), bound_x(x), bound_y(y), bound_w(w), bound_h(h), state(0) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, 255);
		R = dis(gen);
		G = dis(gen);
		B = dis(gen);
		showframe = ID_SHOW_TME;
		detect_finish = false;
	}

	int getId() const { return pid; }
	int getBoundX() const { return bound_x; }
	int getBoundY() const { return bound_y; }
	int getBoundW() const { return bound_w; }
	int getBoundH() const { return bound_h; }
	int getX() const { return centre_x; }
	int getY() const { return centre_y; }
	cv::Scalar getRGB() const { return cv::Scalar(B, G, R); }
	void updateCoords(int xn, int yn, int x, int y, int w, int h) {
		centre_x = xn;
		centre_y = yn;
		bound_x = x;
		bound_y = y;
		bound_w = w;
		bound_h = h;
		tracks.push_back(cv::Point(centre_x, centre_y));
	}

	const std::vector<cv::Point>& getTracks() const { return tracks; }

	void savePic(const cv::Mat& frame) {
		if (bound_x >= 0 && bound_y >= 0 && bound_x + bound_w <= frame.cols && bound_y + bound_h <= frame.rows) {
			sample = frame(cv::Rect(bound_x, bound_y, bound_w, bound_h)).clone();
			std::string path = "F:/image" + std::to_string(pid) + ".jpg";
			cv::imwrite(path, sample);

		}
	}

	std::vector<Defect> defectDetect() {
		if (sample.empty() || detect_finish) return {};

		cv::Mat gray, thresh;
		cv::cvtColor(sample, gray, cv::COLOR_BGR2GRAY);
		cv::threshold(gray, thresh, 135, 250, cv::THRESH_BINARY);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

		for (const auto& cnt : contours) {
			double area = cv::contourArea(cnt);
			if (area > 100 && area <= 1500) {
				cv::Rect rect = cv::boundingRect(cnt);
				cv::Mat roi = sample(rect).clone();
				cv::Mat roi_gray, roi_thresh;
				cv::cvtColor(roi, roi_gray, cv::COLOR_BGR2GRAY);
				cv::threshold(roi_gray, roi_thresh, 135, 250, cv::THRESH_BINARY_INV);

				std::vector<std::vector<cv::Point>> roi_contours;
				cv::findContours(roi_thresh, roi_contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

				cv::Mat mask = cv::Mat::zeros(roi_gray.size(), roi_gray.type());
				cv::drawContours(mask, roi_contours, -1, cv::Scalar(255), cv::FILLED);

				cv::Mat result;
				cv::bitwise_and(roi_gray, mask, result);
				cv::Mat hist;
				int histSize = 256;
				float range[] = { 0, 256 };
				const float* histRange = { range };
				cv::calcHist(&result, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

				double total = cv::sum(hist)[0];
				if (total > 0) {
					hist /= total;
				}

				double hist_sum_scratch = 0.0;
				double hist_sum_blot = 0.0;
				for (int i = 75; i < 135; ++i) {
					hist_sum_scratch += hist.at<float>(i);
				}
				for (int i = 15; i < 90; ++i) {
					hist_sum_blot += hist.at<float>(i);
				}

				if (hist_sum_scratch > 0.1) {
					defects.emplace_back(1, rect.x, rect.y, rect.width, rect.height);
					state = 1;
				}
				if (hist_sum_blot > 0.1) {
					defects.emplace_back(2, rect.x, rect.y, rect.width, rect.height);
					state = 2;
				}
			}
		}

		if (state == 1) {
			scratch_sum++;
		}
		else if (state == 2) {
			blot_sum++;
		}
		detect_finish = true;
		return defects;
	}

	std::vector<Defect> defects;
	int state;

private:
	int pid;
	int centre_x, centre_y;
	int bound_x, bound_y, bound_w, bound_h;
	int R, G, B;
	std::vector<cv::Point> tracks;
	cv::Mat sample;
};
int Product::scratch_sum = 0;
int Product::blot_sum = 0;
int Product::count = 0;

int main() {
	cv::VideoCapture cap("   ");
	if (!cap.isOpened()) {
		std::cerr << "无法打开视频文件" << std::endl;
		return -1;
	}

	int font = cv::FONT_HERSHEY_SIMPLEX;
	std::vector<Product> products;
	int pid = 1;
	const int areaTh = 18000;

	while (true) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty()) {
			std::cout << "EOF" << std::endl;
			break;
		}

		cv::Mat img = frame.clone();
		cv::Mat gray, thresh;
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		cv::threshold(gray, thresh, 127, 255, cv::THRESH_BINARY);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		for (const auto& cnt : contours) {
			double area = cv::contourArea(cnt);
			if (area > areaTh) {
				cv::Moments M = cv::moments(cnt);
				int cx = static_cast<int>(M.m10 / M.m00);
				int cy = static_cast<int>(M.m01 / M.m00);
				cv::Rect rect = cv::boundingRect(cnt);
				int x = rect.x, y = rect.y, w = rect.width, h = rect.height;

				bool newProduct = true;
				if (cx > 100) {
					for (auto& p : products) {
						if (abs(cx - p.getX()) <= 25 && abs(cy - p.getY()) <= 25) {
							newProduct = false;
							p.updateCoords(cx, cy, x, y, w, h);
							break;
						}
					}
					if (newProduct) {
						Product p(pid, cx, cy, x, y, w, h);
						p.savePic(frame);
						p.defects = p.defectDetect();
						products.push_back(p);
						Product::count = pid;
						pid++;
					}
				}
				cv::circle(img, cv::Point(cx, cy), 5, cv::Scalar(0, 0, 255), -1);
				cv::rectangle(img, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 255, 0), 2);
			}
		}

		for (auto& p : products) {
			if (p.getX() <= 600 && p.showframe > 0) {
				cv::putText(img, std::to_string(p.getId()), cv::Point(p.getX(), p.getY()),
					font, 1.0, p.getRGB(), 1, cv::LINE_AA);
				p.showframe--;
				for (const auto& defect : p.defects) {
					int dx = p.getBoundX() + defect.getX();
					int dy = p.getBoundY() + defect.getY();
					int dw = defect.getW() + 5;
					int dh = defect.getH() + 5;
					if (defect.getState() == 1) {
						cv::rectangle(img, cv::Point(dx, dy), cv::Point(dx + dw, dy + dh),
							cv::Scalar(0, 255, 255), 1);
					}
					else if (defect.getState() == 2) {
						cv::rectangle(img, cv::Point(dx, dy), cv::Point(dx + dw, dy + dh),
							cv::Scalar(255, 255, 0), 1);
					}
				}
			}
			cv::putText(img, "sum:" + std::to_string(Product::count), cv::Point(10, 30),
				font, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
			cv::putText(img, "scratch_sum:" + std::to_string(Product::scratch_sum), cv::Point(10, 50),
				font, 0.7, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
			cv::putText(img, "blot_sum:" + std::to_string(Product::blot_sum), cv::Point(10, 70),
				font, 0.7, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
		}

		cv::imshow("test", img);
		int k = cv::waitKey(10) & 0xff;
		if (k == 27) break;
	}

	cv::destroyAllWindows();
	return 0;
}