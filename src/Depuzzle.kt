import org.apache.commons.math3.ml.clustering.CentroidCluster
import org.apache.commons.math3.ml.clustering.DoublePoint
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer
import org.opencv.calib3d.Calib3d.findHomography
import org.opencv.core.*
import org.opencv.core.Core.perspectiveTransform
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
    HoughLines(edges, lines, 10.0, 10 * PI / 180, threshold)
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
//TODO: Figure out if I should be using Point()s intead of DoublePoint()s
/*
h is the height of the original image, which is 2 * d for d the diagonal, i.e., 2 * sqrt(img.rows()^2+img.cols()^2)
newSize shouldn't be changed atm.
 */
fun normalizeHLine(r : Double, theta : Double, h : Double, newSize : Int = 180) : DoublePoint
        = DoublePoint(doubleArrayOf(r / h * newSize, theta * newSize * 2.0 / PI))

fun normalizeHSpace(lines : Mat, h : Double, newSize : Int = 180) : List<DoublePoint>
        = pointList(lines).map { normalizeHLine(it.point[0], it.point[1], h, newSize) }

fun denormalizeHLine(r : Double, theta : Double, h : Double, normalized : Int = 180) : DoublePoint
        = DoublePoint(doubleArrayOf(r / normalized.toDouble() * h, theta * PI / (normalized * 2.0)))

fun denormalizeHSpace(lines : List<DoublePoint>, h : Double, normalized : Int = 180) : List<DoublePoint>
        = lines.map { denormalizeHLine(it.point[0], it.point[1], h, normalized) }

fun pointList(lines : Mat) : MutableList<DoublePoint> {
    val lineArray = mutableListOf<DoublePoint>()
    for (i in 0 until lines.rows()) lineArray.add(DoublePoint(lines[i, 0]))
    return lineArray
}

/*
fun easyPointList(lines : Mat) : List<DoublePoint> {
    val linesMat = lines as MatOfPoint2f
    return linesMat.toList().map { it.toDoublePoint() }
}

fun pointList(lines : Mat) : List<DoublePoint> = (lines as MatOfPoint2f).toList().map { it.toDoublePoint() }
 */

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
    val size : Double
        get() = pt.dist(center)
}

/*
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
 */
fun meanDist(pts : List<ClusterPt>) : Double = pts.map { it.size }.average()
fun meanDist(pts : CentroidCluster<DoublePoint>) : Double
        = meanDist(pts.points.map { ClusterPt(it, DoublePoint(pts.center.point)) })

fun pruneCP(pts : List<ClusterPt>, cutoff : Double) : List<ClusterPt> = pts.filter { it.size <= cutoff }
fun pruneCP(pts : CentroidCluster<DoublePoint>, cutoff : Double) : List<ClusterPt>
        = pruneCP(pts.points.map { ClusterPt(it, DoublePoint(pts.center.point)) }, cutoff)

fun prune(pts : List<ClusterPt>, cutoff : Double) : List<DoublePoint> = pruneCP(pts, cutoff).map { it.pt }
fun prune(pts : CentroidCluster<DoublePoint>, cutoff : Double) : List<DoublePoint> = pruneCP(pts, cutoff).map { it.pt }


fun genIntGrid(width : Int, height : Int) : List<DoublePoint> {
    val l = mutableListOf<DoublePoint>()
    for (i in 0..width) {
        for (j in 0..height) {
            l.add(DoublePoint(intArrayOf(i, j)))
        }
    }
    return l
}

fun genGrid(width : Double, height : Double) : List<DoublePoint> {
    return genIntGrid(9, 9).map {
        DoublePoint(doubleArrayOf(it.point[0] * width / 9.0, it.point[1] * height / 9.0))
    }
}

fun sortLines(lines : List<DoublePoint>) = lines.sortedBy { it.point[0] }

fun Point.toDoublePoint() : DoublePoint = DoublePoint(doubleArrayOf(this.x, this.y))
fun DoublePoint.toPoint() : Point = Point(this.point[0], this.point[1])

fun intersections(lines1 : List<DoublePoint>, lines2: List<DoublePoint>) : List<DoublePoint> {
    val pts = mutableListOf<DoublePoint>()
    for (l1 in lines1) {
        for (l2 in lines2) {
            pts.add(intersection(l1.point[0], l1.point[1], l2.point[0], l2.point[1]).toDoublePoint())
        }
    }
    return pts
}
/*
calculates the point nearest to each in the lattice of ideal intersections.
 */
fun gridFit(grid : List<DoublePoint>, pts : List<DoublePoint>) : HashMap<DoublePoint, DoublePoint> {
    val fits = HashMap<DoublePoint, DoublePoint>(grid.size)
    for (pt in pts) {
        for (gridPt in grid) {
            if (fits.containsKey(gridPt)) {
                val cand = fits[gridPt]!!
                if (pt.dist(gridPt) < cand.dist(gridPt)) fits.put(gridPt, pt)
            } else fits.put(gridPt, pt)
        }
    }
    return fits
}

fun fitEval(fits : HashMap<DoublePoint, DoublePoint>,
            eval : (HashMap<DoublePoint, DoublePoint>) -> Double
                = { it.map { it.key.dist(it.value) }.sum() }
            ) : Double {
    return eval(fits)
}

fun calibrateProjection(corners : List<DoublePoint>, width : Double, height : Double) : Mat {
    val cornerMat = MatOfPoint2f(*(corners.map { it.toPoint() }).toTypedArray())
    val idealCorners = MatOfPoint2f(
        Point(0.0, 0.0),
        Point(width, 0.0),
        Point(0.0, height),
        Point(width, height))
    return findHomography(cornerMat, idealCorners)
}

fun calibrateProjectionInverse(corners : List<DoublePoint>, width : Double, height : Double) : Mat {
    val cornerMat = MatOfPoint2f(*(corners.map { it.toPoint() }).toTypedArray())
    val idealCorners = MatOfPoint2f(
        Point(0.0, 0.0),
        Point(0.0, width),
        Point(height, 0.0),
        Point(width, height))
    return findHomography(idealCorners, cornerMat)
}

fun transformPoints(pts : List<DoublePoint>, homography : Mat) : List<DoublePoint> {
    val ptMat = MatOfPoint2f(*(pts.map { it.toPoint() }).toTypedArray())
    val transMat = Mat()
    perspectiveTransform(ptMat, transMat, homography)
    return pointList(transMat)
}

/*
Supposing lines1 and lines2 contain our respective clusters, this should be their outermost corner points.
 */
fun findCorners(lines1 : List<DoublePoint>, lines2 : List<DoublePoint>) : List<DoublePoint> {
    return intersections(listOf<DoublePoint>(lines1[0], lines1[lines1.lastIndex]),
                         listOf<DoublePoint>(lines2[0], lines2[lines2.lastIndex]))
}

fun orientCorners(corners : List<DoublePoint>) : List<DoublePoint> {
    val firstSort = corners.sortedBy { it.point[1] }
    if (firstSort[1].point[1] < firstSort[2].point[1])
        return listOf(firstSort[1], firstSort[0], firstSort[3], firstSort[2])
    else return firstSort
}

fun calculateGridFit(list1 : List<DoublePoint>,
                     list2 : List<DoublePoint>,
                     eval : (HashMap<DoublePoint, DoublePoint>) -> Double
                        = { it.map { it.key.dist(it.value) }.sum() }
                     ) : Double {
    val corners = orientCorners(findCorners(list1, list2))
    val intersectionPoints = intersections(list1, list2)
    val idealGrid = genGrid(180.0, 180.0)
    val homography = calibrateProjection(corners, 180.0, 180.0)
    val transPoints = transformPoints(intersectionPoints, homography)
    return fitEval(gridFit(idealGrid, transPoints), eval)
}

/*
returns the corners of the best fit
 */
fun findBestFit(list1 : List<DoublePoint>, list2 : List<DoublePoint>) : List<DoublePoint> {
    var bestFitMetric : Double = Double.MAX_VALUE
    var bestValues : IntArray = intArrayOf()
    for (i1 in 0 until (list1.lastIndex - 8)) {
        for (i2 in list1.lastIndex downTo 8) {
            for (j1 in 0 until (list2.lastIndex - 8)) {
                for (j2 in list2.lastIndex downTo 8) { //efficiency!
                    val metric = calculateGridFit(list1.subList(i1, i2), list2.subList(j1, j2))
                    if (metric < bestFitMetric) {
                        bestFitMetric = metric
                        bestValues = intArrayOf(i1, i2, j1, j2)
                    }
                }
            }
        }
    }
    return orientCorners(findCorners(list1.subList(bestValues[0], bestValues[1]),
                                     list2.subList(bestValues[2], bestValues[3])))
}
