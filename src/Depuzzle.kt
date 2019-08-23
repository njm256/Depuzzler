import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.highgui.HighGui.*
import org.opencv.imgcodecs.Imgcodecs.imread
import org.opencv.imgcodecs.Imgcodecs.imwrite
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.*

fun detectEdges() {
    var img = imread("board.jpg")
    var imgGrey = Mat()
    cvtColor(img, imgGrey, COLOR_BGR2GRAY)
    var edges = Mat()
    Canny(imgGrey, edges, 100.0,300.0)
    var lines = Mat()
    for (i in 5..500 step 10) {
        HoughLinesP(edges, lines, 1.0, kotlin.math.PI/180, i)
        println("thresh: " + i + " lines: " + lines.rows())
    }
    /*

    println(edges.width())
    println(edges.height())
    HoughLinesP(edges, lines, 1.0, kotlin.math.PI/180, 150)
    for (i in 0..3) println(lines[0, 0][i])

    for (i in 0 until lines.rows()) {
            line(
                img,
                Point(lines[i, 0][0], lines[i, 0][1]),
                Point(lines[i, 0][2], lines[i, 0][3]),
                Scalar(10.0, 10.0, 250.0),
                15
            )
    }

    for (i in 0..150 step 50) {
        for (j in 300..600 step 25) {
            Canny(imgGrey, edges, i.toDouble(), j.toDouble())
            imwrite("board_lines_" + i + "_" + j + ".jpg", edges)
        }
    }
    */
    imwrite("board_lines.jpg", img)

}