import org.opencv.core.*
import org.opencv.highgui.HighGui.*
import org.opencv.imgcodecs.Imgcodecs.imread
import org.opencv.imgcodecs.Imgcodecs.imwrite
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.*
import kotlin.math.*

fun detectEdges() {
    val img = imread("board.jpg")
    /*
    var imgGrey = Mat()
    cvtColor(img, imgGrey, COLOR_BGR2GRAY)
    var edges = Mat()
    blur(imgGrey, imgGrey, Size(3.0, 3.0))
    Canny(imgGrey, edges, 255.0 / 3.0, 255.0)
    var lines = Mat()
    for (i in 185..385 step 10) {
        HoughLines(edges, lines, 1.0, kotlin.math.PI / 180, i)
        println("thresh: " + i + " lines: " + lines.rows())
        var newImg = img.clone()
        for (j in 0 until lines.rows()) {
            houghLine(newImg, lines[j, 0][0], lines[j, 0][1], Scalar(0.0, 0.0, 255.0), 15)
        }
        imwrite("board_lines_" + i + ".jpg", newImg)
    }
    */
    for (i in 0 until img.cols() step 100) {
        line(img, Point(i.toDouble(), 0.0), Point(i.toDouble(), img.rows() - 1.0), Scalar(0.0, 0.0, 255.0), 15)
    }
    imwrite("board_lines.jpg", img)
}

fun houghLine(img: Mat, r: Double, theta: Double, color: Scalar, thickness: Int): Unit {
    val p1 : Point
    val p2 : Point
    val xint = r / cos(theta)
    val yint = r / sin(theta)
    if (theta < PI / 2.0) {
        if (yint < 0) {
            println("r: {r}, theta: {theta}, yint: {yint}")
            p1 = Point(0.0, 0.0)
        } else if (yint < img.rows()) {
            p1 = Point(0.0, yint)
        } else { // yint >= img.rows()
            p1 = Point(xint - (img.rows() - 1.0) * tan(theta), img.rows() - 1.0)
        }

        if (xint < 0.0) {
            println("r: {r}, theta: {theta}, xint: {xint}")
            p2 = Point(0.0, 0.0)
        } else if (xint < img.cols()) {
            p2 = Point(xint, 0.0)
        } else { // xint >= img.cols()
            p2 = Point(img.cols() - 1.0, yint - (img.cols() - 1.0)/tan(theta))
        }
    } else {
        if (yint < 0) {
            p1 = Point(xint ,0.0)
        } else if (yint < img.rows())
    }

    line(img, p1, p2, color, thickness)
}