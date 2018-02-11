using namespace cv;

bool is_flow_correct(Point2f u);

Vec3b compute_color(float fx, float fy);

void draw_optical_flow(const Mat_<Point2f>& flow, Mat & dst, float maxmotion=-1);
