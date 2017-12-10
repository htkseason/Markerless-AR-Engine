import java.io.IOException;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import pers.season.vml.ar.CameraData;
import pers.season.vml.ar.Engine3D;
import pers.season.vml.ar.TemplateDetector;
import pers.season.vml.ar.MotionFilter;
import pers.season.vml.util.*;

public final class Entrance {

	static {
		System.loadLibrary("lib/opencv_java320_x64");
	}

	public static void main(String[] args) throws IOException {
		Mat pic = new Mat();
		VideoCapture vc = new VideoCapture();
		vc.open(0);

		Mat template = Imgcodecs.imread("./target2.jpg");
		

		TemplateDetector td = new TemplateDetector();
		td.setTemplate(template);
		

		Engine3D e3 = new Engine3D(CameraData.MY_CAMERA, template);
		MotionFilter tvecFilter = new MotionFilter(3, 0.25);
		MotionFilter rvecFilter = new MotionFilter(3, 0.25);
		ImUtils.imshow(template);
		while (true) {
			vc.read(pic);

			
			Mat homo = td.findHomo(pic, true);

			if (homo != null) {
				Mat quad = td.getQuadFromHomo(homo);

				for (int i = 0; i < quad.total(); i++) {
					Imgproc.line(pic, new Point(quad.get(i, 0)[0], quad.get(i, 0)[1]),
							new Point(quad.get((int) ((i + 1) % quad.total()), 0)[0],
									quad.get((int) ((i + 1) % quad.total()), 0)[1]),
							new Scalar(0, 255, 0), 3);
				}

				Mat rvec = new Mat(), tvec = new Mat();
				td.solvePnp(homo, CameraData.MY_CAMERA, rvec, tvec);
				e3.update(pic, rvecFilter.next(rvec), tvecFilter.next(tvec));
			} else {
				e3.update(pic, null, null);
				rvecFilter.reset();
				tvecFilter.reset();
			}

		}

	}

}
