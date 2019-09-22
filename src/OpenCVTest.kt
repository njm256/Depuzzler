import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs.imread
import org.opencv.imgcodecs.Imgcodecs.imwrite
import kotlin.math.pow
import kotlin.math.sqrt

fun main(args : Array<String>) {
    println("Welcome to OpenCV " + Core.VERSION)
    println(System.getProperty("java.library.path"))
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val img : Mat = imread("imgs/board_3.jpg")
    val lines = getHoughLines(img, 480)
    val h = sqrt(img.rows().toDouble().pow(2) + img.cols().toDouble().pow(2))
    val nlines = normalizeHSpace(lines, h)
    val clusters = findClusters(nlines)
    for (nl in pruneCluster(clusters[0], 15)) {
        val l = denormalizeHLine(nl.point[0], nl.point[1], h)
        houghLine(img, l.point[0], l.point[1], Scalar(0.0, 0.0, 255.0), 10)
    }
    for (nl in pruneCluster(clusters[1], 15)) {
        val l = denormalizeHLine(nl.point[0], nl.point[1], h)
        houghLine(img, l.point[0], l.point[1], Scalar(255.0, 0.0, 0.0), 10)
    }
    imwrite("imgs/board3_lines.jpg", img)
}