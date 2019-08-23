import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat

fun main(args : Array<String>) {
    println("Welcome to OpenCV " + Core.VERSION)
    println(System.getProperty("java.library.path"))
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val m = Mat.eye(3, 3, CvType.CV_8UC1)
    println("m = " + m.dump())
    detectEdges()
}