import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs.imread
import org.opencv.imgcodecs.Imgcodecs.imwrite
import org.opencv.imgproc.Imgproc.circle
import kotlin.math.pow
import kotlin.math.sqrt

fun main(args: Array<String>) {
    println("Welcome to OpenCV " + Core.VERSION)
    println(System.getProperty("java.library.path"))
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val img: Mat = imread("imgs/board_3.jpg")
    val lines: Mat
    lines = getHoughLines(img, 600)
    val h = sqrt(img.rows().toDouble().pow(2) + img.cols().toDouble().pow(2))
    val nlines = normalizeHSpace(lines, h)
    val clusters = findClusters(nlines)
    val prunedClusters = clusters.map { prune(it, meanDist(it) * 1.2) }
    val trueClusters = prunedClusters.map { denormalizeHSpace(it, h) }
    for (i in 0..1) {
        for (l in trueClusters[i]) {
            houghLine(img, l.point[0], l.point[1], Scalar(255.0 * i, 0.0, 255.0 * (1 - i)), 10)
        }
    }
    val corners = findBestFit(trueClusters[0], trueClusters[1])
    val idealGrid = genGrid(90.0, 90.0)
    val homography = calibrateProjectionInverse(corners, 90.0, 90.0)
    val mappedGrid = transformPoints(idealGrid, homography)
    mappedGrid.map { it.toPoint() }.map { circle(img, it, 5, Scalar(0.0, 255.0, 0.0)) }
    imwrite("imgs/board3_intersections.jpg", img)
}