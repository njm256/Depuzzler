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
    var img : Mat
/*
    for (i in 1..5) {
        img = imread("imgs/board_${i}.jpg")
        for (j in 1..8) {
            blurGrey(img, j.toDouble(), "imgs/board_${i}")
        }
    }

    img = imread("imgs/board_1.jpg")
    detectEdgesP(img, "imgs/board_1")
    */
    img = imread("imgs/bwboard_empty.jpg")
    //detectEdges(img, "imgs/bwboard")
    val lines = getHoughLines(img, 285)
    val h = sqrt(img.rows().toDouble().pow(2) + img.cols().toDouble().pow(2))
    //val nlines = normalizeHSpace(lines, h)
    println("finding clusters")
    //val clusters = findClusters(nlines)
    val clusters = findClusters(lines)
    println("found ${clusters.size} clusters")
    for (nl in clusters[0].points) {
        //val l = denormalizeHLine(nl.point[0], nl.point[1], h)
        val l = nl
        println("drawing line in cluster 0: ${l.point[0]}, ${l.point[1]}")
        houghLine(img, l.point[0], l.point[1], Scalar(0.0, 0.0, 255.0), 10)
    }
    for (nl in clusters[1].points) {
        val l = nl
        //val l = denormalizeHLine(nl.point[0], nl.point[1], h)
        println("drawing line in cluster 1: ${l.point[0]}, ${l.point[1]}")
        houghLine(img, l.point[0], l.point[1], Scalar(255.0, 0.0, 0.0), 10)
    }
    imwrite("imgs/bwboard.jpg", img)
}