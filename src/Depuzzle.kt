import org.apache.commons.math3.ml.clustering.CentroidCluster
import org.apache.commons.math3.ml.clustering.DoublePoint
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs.imwrite
import org.opencv.imgproc.Imgproc.*
import kotlin.math.*

fun detectEdges(img : Mat, name : String) {
    var lines : Mat
    for (i in 185..385 step 10) {
        lines = getHoughLines(img, i)
        println("${name} thresh: " + i + " lines: " + lines.rows())
        var newImg = img.clone()
        for (j in 0 until lines.rows()) {
            houghLine(newImg, lines[j, 0][0], lines[j, 0][1], Scalar(0.0, 0.0, 255.0), 15)
        }
        imwrite("${name}_$i.jpg", newImg)
    }
}

fun detectEdgesP(img : Mat, name : String) {
    var lines : Mat
    for (i in 185..385 step 10) {
        lines = getHoughLinesP(img, i)
        println("${name} thresh: " + i + " lines: " + lines.rows())
        var newImg = img.clone()
        for (j in 0 until lines.rows()) {
            houghLineP(newImg, lines[j, 0][0], lines[j, 0][1], lines[j, 0][2], lines[j, 0][3], Scalar(0.0, 0.0, 255.0), 15)
        }
        imwrite("${name}_$i.jpg", newImg)
    }
}

/*
output is an n x 1 matrix whose entries are tuples of coordinates... i.e.,
    [ [ x1, y1] ,
      [ x2, y2] ,
      ...
      ...
      [ xn, yn] ]
      ...sigh.
 */
fun getHoughLines(img : Mat, threshold : Int) : Mat {
    val imgGrey = Mat()
    cvtColor(img, imgGrey, COLOR_BGR2GRAY)
    val edges = Mat()
    blur(imgGrey, imgGrey, Size(2.0, 2.0))
    Canny(imgGrey, edges, 255.0 / 3.0, 255.0)
    val lines = Mat()
    HoughLines(edges, lines, 1.0, PI / 180, threshold)
    return lines
}

fun getHoughLinesP(img : Mat, threshold : Int) : Mat {
    val imgGrey = Mat()
    cvtColor(img, imgGrey, COLOR_BGR2GRAY)
    val edges = Mat()
    blur(imgGrey, imgGrey, Size(3.0, 3.0))
    Canny(imgGrey, edges, 255.0 / 3.0, 255.0)
    val lines = Mat()
    val len = minOf(img.rows(), img.cols()).toDouble()
    HoughLinesP(imgGrey, lines, 3.0, 2.0 * PI / 180, threshold, len / 2.0, 5.0)
    return lines
}

/*
draws a line of the form (r, theta) on img. Couldn't find a simpler way to 'deHough', ugh.
 */
fun houghLine(img: Mat, r: Double, theta: Double, color: Scalar, thickness: Int) {
    val p1: Point
    val p2: Point
    val xint = r / cos(theta)
    val yint = r / sin(theta)
    if (theta < PI / 2.0) {
        if (yint < 0.0) {
            println("r: {r}, theta: {theta}, yint: {yint}")
            p1 = Point(0.0, 0.0)
        } else if (yint < img.rows()) {
            p1 = Point(0.0, yint)
        } else { // yint >= img.rows()
            p1 = Point(xint - (img.rows() - 1.0) * tan(theta), img.rows() - 1.0)
        }

        if (xint < 0.0) {
            println("r: ${r}, theta: ${theta}, xint: ${xint}")
            p2 = Point(0.0, 0.0)
        } else if (xint < img.cols()) {
            p2 = Point(xint, 0.0)
        } else { // xint >= img.cols()
            p2 = Point(img.cols() - 1.0, yint - (img.cols() - 1.0) / tan(theta))
        }
    } else {
        if (yint < 0) {
            p1 = Point(xint, 0.0)
        } else if (yint < img.rows()) {
            p1 = Point(0.0, yint)
        } else { //yint >= img.rows()
            println("r: ${r}, theta: ${theta}, yint: ${yint}")
            p1 = Point(0.0, 0.0)
        }

        val oyint = yint - (img.cols() - 1.0) / tan(theta)
        if (oyint < 0) {
            println("r: ${r}, theta: ${theta}, wtf.")
            p2 = Point(0.0, 0.0)
        } else if (oyint < img.rows()) {
            p2 = Point(img.cols() - 1.0, oyint)
        } else {
            p2 = Point(xint - tan(theta) * (img.cols() - 1.0), img.rows() - 1.0)
        }
    }
    line(img, p1, p2, color, thickness)
}

fun houghLineP(img : Mat, x1 : Double, y1 : Double, x2 : Double, y2 : Double, color : Scalar, thickness : Int)
    = line(img, Point(x1, y1), Point(x2, y2), color, thickness)


fun intersection(r1 : Double, theta1 : Double, r2 : Double, theta2 : Double)
    = Point((r1 * sin(theta2) - r2 * sin(theta1)) / sin(theta2 - theta1)
           ,(r1 * cos(theta2) + r2 * cos(theta1)) / sin(theta1 - theta2))

fun blurGrey(img : Mat, kernel : Double, str : String) {
    val imgGrey = Mat()
    val edges = Mat()
    cvtColor(img, imgGrey, COLOR_BGR2GRAY)
    Canny(imgGrey, edges, 255.0/3.0, 255.0)
    imwrite("${str}_${kernel}_edges.jpg", edges)
}

/*
h is the height of the original image, which is 2 * d for d the diagonal, i.e., 2 * sqrt(img.rows()^2+img.cols()^2)
newSize shouldn't be changed atm.
 */
fun normalizeHLine(r : Double, theta : Double, h : Double, newSize : Int = 180) : DoublePoint
    = DoublePoint(doubleArrayOf(r / h * newSize, theta * newSize / PI))

fun normalizeHSpace(lines : Mat, h : Double, newSize : Int = 180) : List<DoublePoint>
    = pointList(lines).map { normalizeHLine(it.point[0], it.point[1], h, newSize) }

fun denormalizeHLine(r : Double, theta : Double, h : Double, normalized : Int = 180) : DoublePoint {
    println("denormalized ${r}, ${theta} to ${r / normalized.toDouble() * h}, ${theta * PI / normalized}")
    return DoublePoint(doubleArrayOf(r / normalized.toDouble() * h, theta * PI / normalized))
}

fun denormalizeHSpace(lines : List<DoublePoint>, h : Double, normalized : Int = 180) : List<DoublePoint>
    = lines.map { denormalizeHLine(it.point[0], it.point[1], h, normalized) }

fun pointList(lines : Mat) : MutableList<DoublePoint> {
    val lineArray = mutableListOf<DoublePoint>()
    for (i in 0 until lines.rows()) lineArray.add(DoublePoint(lines[i, 0]))
    return lineArray
}

fun findClusters(lines : List<DoublePoint>) : MutableList<CentroidCluster<DoublePoint>>
        = KMeansPlusPlusClusterer<DoublePoint>(2).cluster(lines)

fun DoublePoint.dist(other : DoublePoint) : Double
    = sqrt((this.point[0] - other.point[0]).pow(2) + (this.point[1] - other.point[1]).pow(2))

fun sgn(x : Double) : Int = when {
        x == 0.0 -> 0
        x > 0.0 -> 1
        else -> -1
    }

data class ClusterPt(val pt  : DoublePoint, val center : DoublePoint) : Comparable<ClusterPt> {
    override operator fun compareTo(other : ClusterPt) = sgn(pt.dist(center) - other.pt.dist(other.center))
}

fun prune(pts : MutableList<CentroidCluster<DoublePoint>>, n : Int): List<DoublePoint> {
    return pts.flatMap { cluster -> cluster.points.map {
        ClusterPt(pt = it, center = DoublePoint(cluster.center.point))
    } }.sorted().slice(0 until n).map { it.pt }
}

fun pruneCluster(cluster : CentroidCluster<DoublePoint>, n : Int) : List<DoublePoint>
    = cluster.points.sortedBy { it.dist(DoublePoint(cluster.center.point)) }.slice(0 until n)

fun pruneLimit(clusters : MutableList<CentroidCluster<DoublePoint>>, n : Int, lim : Int) : List<DoublePoint> {
    val pts =  clusters.map { cluster -> cluster.points.map {
        ClusterPt(pt = it, center = DoublePoint(cluster.center.point)) } }

    var pruning = pts.map { it.size }.toMutableList()

    var ptlist = pts.flatten().sortedDescending()
    for (i in 0 until n) {
        val index = pts.indexOfFirst { it[0].center == ptlist[0].center }
        if (pruning[index] <= lim) {
            break
        }
        pruning[index] = pruning[index] - 1
        ptlist = ptlist.slice(1 until ptlist.size)
    }
    return ptlist.map { it.pt }
}