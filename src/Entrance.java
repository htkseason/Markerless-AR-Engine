

import java.io.IOException;
import java.util.UUID;

import javax.swing.JFrame;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import pers.season.vml.ar.CameraData;
import pers.season.vml.ar.Engine3D;
import pers.season.vml.ar.FeatureDM;
import pers.season.vml.ar.MotionFilter;
import pers.season.vml.statistics.appearance.AppearanceFitting;
import pers.season.vml.statistics.appearance.AppearanceModel;
import pers.season.vml.statistics.appearance.AppearanceModelTrain;
import pers.season.vml.statistics.regressor.LearningParams;
import pers.season.vml.statistics.regressor.RegressorTrain;
import pers.season.vml.statistics.shape.ShapeInstance;
import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.statistics.shape.ShapeModelTrain;
import pers.season.vml.statistics.texture.TextureInstance;
import pers.season.vml.statistics.texture.TextureModel;
import pers.season.vml.statistics.texture.TextureModelTrain;
import pers.season.vml.util.*;

public final class Entrance {

	static {
		System.loadLibrary("lib/opencv_java320_x64");
	}

	public static void main(String[] args) throws IOException {
		Mat pic = new Mat();
		VideoCapture vc = new VideoCapture();
		vc.open(0);

		Mat template = Imgcodecs.imread("./miku.jpg");

		FeatureDM fdm = new FeatureDM();
		fdm.setTemplate(template);

		Engine3D e3 = new Engine3D(CameraData.MY_CAMERA, template);
		MotionFilter tvecFilter = new MotionFilter(3, 0.25);
		MotionFilter rvecFilter = new MotionFilter(3, 0.25);
		while (true) {
			vc.read(pic);
			Mat homo = fdm.findHomo(pic, true);
			Mat rvec = new Mat(), tvec = new Mat();
			if (homo != null) {
			
				Mat quad = fdm.getQuadFromHomo(homo);

				for (int i = 0; i < quad.total(); i++) {
					Imgproc.line(pic, new Point(quad.get(i, 0)[0], quad.get(i, 0)[1]),
							new Point(quad.get((int) ((i + 1) % quad.total()), 0)[0],
									quad.get((int) ((i + 1) % quad.total()), 0)[1]),
							new Scalar(0, 255, 0), 3);
				}
				

				fdm.solvePnp(homo, CameraData.MY_CAMERA, rvec, tvec);

				e3.update(pic, rvecFilter.next(rvec), tvecFilter.next(tvec));
			} else {
				e3.update(pic, null, null);
				rvecFilter.reset();
				tvecFilter.reset();
			}


		}

	}

}
