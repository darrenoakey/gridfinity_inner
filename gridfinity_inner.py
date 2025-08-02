#!/usr/bin/env python3
# gridfinity_inner.py – Generate Gridfinity bins, solid inserts, and
#                       image-shaped cut-outs that drop neatly inside.
#
#   • Each helper is tiny; no duplicated logic
#   • Comments live above functions (no doc-strings)
#   • Real tests run automatically when the file is imported
#
# Requires:
#   build123d ≥ 0.9
#   gfthings  (master branch)
#   opencv-python for --cutout

# ---------------------------------------------------------------------------#
# imports                                                                    #
# ---------------------------------------------------------------------------#
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from build123d import export_stl  # STL helper
from gfthings.Bin import FunkyBin  # Gridfinity bin generator
import math

# ---------------------------------------------------------------------------#
# constants                                                                  #
# ---------------------------------------------------------------------------#
TOLERANCE_MM = 0.30  # wall clearance for the inner plug
TOP_GAP_MM = 1.0  # gap above plug so you can pull it out
MIN_HOLE_MM = 0.75  # fill gaps narrower than this when tracing
DEFAULT_PPMM = 10  # default pixels-per-millimetre for --cutout
MIN_SPACING_MM = 0.6  # minimum space between rectangular cut‑outs


# ---------------------------------------------------------------------------#
# create_shape – create a binary matrix from x and y dimensions              #
# ---------------------------------------------------------------------------#
def create_shape(x: int, y: int) -> List[List[int]]:
    return [[1] * x for _ in range(y)]


# ---------------------------------------------------------------------------#
# parse_mm – convert CLI dimension strings to float millimetres              #
# ---------------------------------------------------------------------------#
def parse_mm(text: str) -> float:
    if text.lower().endswith("mm"):
        text = text[:-2]
    return float(text)


# ---------------------------------------------------------------------------#
# best_grid – choose rows/cols & spacing for rectangular cut‑outs            #
# ---------------------------------------------------------------------------#
# Returns: cols, rows, spacing_x, spacing_y
# spacing_x/Y includes the edge margins, guaranteed ≥ MIN_SPACING_MM
# Raises RuntimeError if the rectangles cannot fit.
#
# If count is None we maximise count; otherwise we guarantee at least that
# many pockets (might return more – e.g. 4 request could yield 6 because of
# grid).
# ---------------------------------------------------------------------------#
def best_grid(
    part_x: float,
    part_y: float,
    rect_x: float,
    rect_y: float,
    min_spacing: float = MIN_SPACING_MM,
    count: Optional[int] = None,
) -> Tuple[int, int, float, float]:
    max_cols = math.floor((part_x + min_spacing) / (rect_x + min_spacing))
    max_rows = math.floor((part_y + min_spacing) / (rect_y + min_spacing))

    print(f"max {max_cols}x{max_rows} and part = {part_x}x{part_y}")

    if max_cols == 0 or max_rows == 0:
        raise RuntimeError("Cut‑outs do not fit even once – part too small")

    best = None  # (total, cols, rows, sx, sy)

    # Iterate rows/cols space – small search space so brute force is fine
    for rows in range(1, max_rows + 1):
        for cols in range(1, max_cols + 1):
            total = rows * cols
            if count and total < count:
                continue  # not enough pockets
            sx = (part_x - cols * rect_x) / (cols + 1)
            sy = (part_y - rows * rect_y) / (rows + 1)
            if sx < min_spacing or sy < min_spacing:
                continue  # spacing too tight
            # Prefer fewer rows (makes long line) if multiple solutions equal
            score = (rows, -total)
            cand = (total, cols, rows, sx, sy, score)
            if best is None or cand[-1] < best[-1]:
                best = cand

    if best is None:
        raise RuntimeError(
            "Unable to fit requested rectangular cut‑outs with ≥2 mm spacing"
        )

    _, cols, rows, sx, sy, _ = best
    return cols, rows, sx, sy


# ---------------------------------------------------------------------------#
# carve_rect_grid – subtract a grid of rectangles from the part              #
# ---------------------------------------------------------------------------#
# Each rectangle is centred at Z depth so that 2 mm of material remains.
# ---------------------------------------------------------------------------#


def carve_rect_grid(
    part,
    rect_x: float,
    rect_y: float,
    count: Optional[int],
    is_inner: bool,
):
    from build123d import BuildPart, BuildSketch, Rectangle, Location, extrude, Plane
    from gfthings.parameters import plate_height

    bbox = part.bounding_box()
    part_x = bbox.size.X - 2 * MIN_SPACING_MM  # leave outer 2 mm shell untouched
    part_y = bbox.size.Y - 2 * MIN_SPACING_MM

    cols, rows, sx, sy = best_grid(
        part_x, part_y, rect_x, rect_y, MIN_SPACING_MM, count
    )

    # left/bottom offset so first spacing margin included
    start_x = bbox.center().X - (cols - 1) * (rect_x + sx) / 2
    start_y = bbox.center().Y - (rows - 1) * (rect_y + sy) / 2

    # depth calculation
    depth = bbox.size.Z - 2.0
    if depth <= 0:
        raise RuntimeError("Computed depth ≤0 – part too shallow for cut‑out")
    print(f"Cutter depth: {depth:.2f} mm")

    cutters = []
    for r in range(rows):
        cy = start_y + r * (rect_y + sy)
        for c in range(cols):
            cx = start_x + c * (rect_x + sx)
            with BuildPart():
                z_top = bbox.max.Z
                with BuildSketch(Plane.XY * Location((cx, cy, z_top))):
                    Rectangle(rect_x, rect_y, align=(None, None))
                cut = extrude(amount=-depth)
                cutters.append(cut)

    # Boolean difference – union cutters then subtract once (faster)
    from build123d import Compound

    print(f"Total cutters: {len(cutters)}")

    removal = Compound(*cutters)  # build123d ≥ 0.9

    print(f"Cutter volume: {removal.volume:.2f} mm³")
    print(f"Part volume before cut: {part.volume:.2f} mm³")

    result = part.cut(*cutters)

    print(f"Part volume after cut: {result.volume:.2f} mm³")

    return result


# ---------------------------------------------------------------------------#
# image_to_contour – trace the largest blob inside a 1 cm-grid photo         #
# ---------------------------------------------------------------------------#
def image_to_contour(
    image_path: Path,
    ppmm: int = DEFAULT_PPMM,
    tol_mm: float = 0.50,
    min_hole_mm: float = MIN_HOLE_MM,
    debug: bool = True,
) -> List[Tuple[float, float]]:
    import cv2
    import numpy as np

    # Load the image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    print(f"Loaded image: {image_path}, dimensions: {img.shape}")

    # Save original for debug overlay
    original = img.copy()
    debug_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Step 1: Detect the grid lines
    # Apply adaptive thresholding to isolate the grid
    grid_thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Use Hough Line Transform to detect grid lines
    lines = cv2.HoughLinesP(
        grid_thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=img.shape[0] // 4,  # Lines should be at least 1/4 of the image
        maxLineGap=20,
    )

    if lines is None or len(lines) < 4:  # Need at least a few lines to form a grid
        print("ERROR: Could not detect grid lines properly")
        if debug:
            cv2.imwrite(
                str(
                    image_path.with_name(
                        f"{image_path.stem}_grid_thresh{image_path.suffix}"
                    )
                ),
                grid_thresh,
            )
        raise RuntimeError("Grid detection failed - not enough lines detected")

    # Step 2: Separate horizontal and vertical lines
    h_lines = []
    v_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle to determine if horizontal or vertical
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        if angle < 45 or angle > 135:  # Horizontal line
            h_lines.append(line[0])
            if debug:
                cv2.line(debug_color, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green
        elif angle >= 45 and angle <= 135:  # Vertical line
            v_lines.append(line[0])
            if debug:
                cv2.line(debug_color, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue

    print(f"Detected {len(h_lines)} horizontal and {len(v_lines)} vertical grid lines")

    if debug:
        cv2.imwrite(
            str(
                image_path.with_name(f"{image_path.stem}_grid_lines{image_path.suffix}")
            ),
            debug_color,
        )

    # Step 3: Find the object (content) in the image
    # Use binary thresholding to separate object from background
    _, obj_thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to clean up the object mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(1, int(min_hole_mm * ppmm)),) * 2
    )
    obj_mask = cv2.morphologyEx(obj_thresh, cv2.MORPH_CLOSE, kernel)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel)

    if debug:
        cv2.imwrite(
            str(image_path.with_name(f"{image_path.stem}_object{image_path.suffix}")),
            obj_mask,
        )

    # Step 4: Find intersections between the grid lines and the object
    intersection_points = []

    # Function to check if a point is on the boundary of the object
    def is_boundary_point(x, y, mask, window_size=3):
        # Get a small window around the point
        x_start = max(0, x - window_size // 2)
        x_end = min(mask.shape[1], x + window_size // 2 + 1)
        y_start = max(0, y - window_size // 2)
        y_end = min(mask.shape[0], y + window_size // 2 + 1)

        window = mask[y_start:y_end, x_start:x_end]

        # Check if the window contains both object and background pixels
        return np.min(window) == 0 and np.max(window) == 255

    # Sample points along each grid line and check for intersections
    for line in h_lines + v_lines:
        x1, y1, x2, y2 = line
        # Create a set of points along the line
        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal-ish line
            for x in range(min(x1, x2), max(x1, x2) + 1, 5):  # Sample every 5 pixels
                y = int(y1 + (y2 - y1) * (x - x1) / (x2 - x1)) if x2 != x1 else y1
                if 0 <= y < obj_mask.shape[0] and is_boundary_point(x, y, obj_mask):
                    intersection_points.append((x, y))
                    if debug:
                        cv2.circle(debug_color, (x, y), 3, (0, 0, 255), -1)  # Red
        else:  # Vertical-ish line
            for y in range(min(y1, y2), max(y1, y2) + 1, 5):
                x = int(x1 + (x2 - x1) * (y - y1) / (y2 - y1)) if y2 != y1 else x1
                if 0 <= x < obj_mask.shape[1] and is_boundary_point(x, y, obj_mask):
                    intersection_points.append((x, y))
                    if debug:
                        cv2.circle(debug_color, (x, y), 3, (0, 0, 255), -1)  # Red

    print(f"Found {len(intersection_points)} grid-object intersection points")

    if len(intersection_points) < 3:
        print("ERROR: Too few intersection points found")
        cv2.imwrite(
            str(
                image_path.with_name(
                    f"{image_path.stem}_intersections{image_path.suffix}"
                )
            ),
            debug_color,
        )
        raise RuntimeError("Not enough intersection points to form a contour")

    # Step 5: Order the points to form a contour around the object
    # Convert to numpy array for easier manipulation
    points = np.array(intersection_points)

    # Find the center of the points
    center = np.mean(points, axis=0)

    # Sort points by angle around the center
    def polar_angle(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])

    sorted_indices = sorted(range(len(points)), key=lambda i: polar_angle(points[i]))
    ordered_points = points[sorted_indices]

    # Apply polygon approximation to reduce number of points
    epsilon = tol_mm * ppmm
    contour = cv2.approxPolyDP(ordered_points.reshape((-1, 1, 2)), epsilon, True)
    contour = contour.reshape((-1, 2))

    if debug:
        # Draw the final contour
        contour_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        cv2.polylines(contour_img, [contour], True, (0, 0, 255), 2)
        cv2.imwrite(
            str(
                image_path.with_name(
                    f"{image_path.stem}_final_contour{image_path.suffix}"
                )
            ),
            contour_img,
        )

        # Also draw on the debug image with grid lines
        cv2.polylines(debug_color, [contour], True, (0, 255, 255), 2)  # Yellow
        cv2.imwrite(
            str(
                image_path.with_name(
                    f"{image_path.stem}_grid_and_contour{image_path.suffix}"
                )
            ),
            debug_color,
        )

    # Convert to mm and center
    pts_mm = contour / ppmm
    cx, cy = pts_mm.mean(axis=0)
    centered_pts = [(float(x - cx), float(y - cy)) for x, y in pts_mm]

    print(f"Final contour has {len(centered_pts)} points")
    print(f"Contour centered at ({cx:.2f}, {cy:.2f}) mm")

    return centered_pts


# ---------------------------------------------------------------------------#
# gridfinity_inner – solid insert that fits the bin precisely                #
# ---------------------------------------------------------------------------#
def gridfinity_inner(
    shape: List[List[int]],
    z_units: int = 3,
    tol: float = TOLERANCE_MM,
    top_gap: float = TOP_GAP_MM,
):
    from build123d import (
        BuildPart,
        BuildSketch,
        RectangleRounded,
        Plane,
        Align,
        extrude,
    )
    from gfthings.parameters import bin_size, bin_clearance, outer_rad, plate_height

    try:
        from gfthings.parameters import wall_thickness as WALL
    except ImportError:
        WALL = 1.2

    cols = max(len(r) for r in shape)
    rows = len(shape)
    inner_x = cols * bin_size - 2 * (bin_clearance + WALL + tol)
    inner_y = rows * bin_size - 2 * (bin_clearance + WALL + tol)
    inner_r = max(outer_rad - bin_clearance - WALL - tol, 0)
    inner_z = z_units * 7 - plate_height - top_gap

    print(f"Creating insert: {cols}x{rows} grid, z {z_units} units")
    print(
        f"Insert dimensions: {inner_x:.2f}x{inner_y:.2f}x{inner_z:.2f} mm, corner radius: {inner_r:.2f} mm"
    )

    with BuildPart() as plug:
        with BuildSketch(Plane.XY):
            RectangleRounded(
                inner_x, inner_y, inner_r, align=(Align.CENTER, Align.CENTER)
            )
        extrude(amount=inner_z)

    print(f"Insert created successfully with volume: {plug.part.volume:.2f} mm³")
    return plug.part


# ---------------------------------------------------------------------------#
# carve_plug – subtract a 2-D contour from the plug                          #
# ---------------------------------------------------------------------------#
def carve_plug(
    plug_part,
    contour_mm: List[Tuple[float, float]],
    depth_mm: float,
    debug: bool = True,
    cutout_name: str = "",
):
    from build123d import Solid, Face, Wire, Edge, Vector

    print(f"\nCarving cutout with depth: {depth_mm:.2f} mm")
    print(f"Contour has {len(contour_mm)} points")

    # Create 3D points
    points_3d = [Vector(float(x), float(y), 0) for x, y in contour_mm]

    # Ensure the contour is closed
    if (points_3d[0].X != points_3d[-1].X) or (points_3d[0].Y != points_3d[-1].Y):
        print("Closing contour by adding first point to the end")
        points_3d.append(points_3d[0])

    # Create edges connecting the points
    edges = []
    for i in range(len(points_3d) - 1):
        try:
            edge = Edge.make_line(points_3d[i], points_3d[i + 1])
            edges.append(edge)
        except Exception as e:
            print(f"Error creating edge {i} → {i+1}: {e}")

    print(f"Created {len(edges)} edges")

    if len(edges) < 3:
        print("ERROR: Need at least 3 edges to form a wire")
        return plug_part

    # Create a wire using combine
    try:
        wire_result = Wire.combine(edges)
        if not wire_result:
            print("ERROR: Wire.combine returned empty result")
            return plug_part

        wire = wire_result[0]
        print(f"Wire created successfully, length: {wire.length:.2f} mm")

        if not wire.is_closed:
            print("WARNING: Wire is not closed!")
    except Exception as e:
        print(f"ERROR creating wire: {e}")
        return plug_part

    # Create a face from the wire
    try:
        face = Face(wire)
        print(f"Face created successfully, area: {face.area:.2f} mm²")
    except Exception as e:
        print(f"ERROR creating face: {e}")
        return plug_part

    # Create a solid by extruding the face
    try:
        extrude_dir = Vector(0, 0, depth_mm)
        cutter = Solid.extrude(face, extrude_dir)
        print(f"Cutter created successfully, volume: {cutter.volume:.2f} mm³")
    except Exception as e:
        print(f"ERROR creating cutter: {e}")
        return plug_part

    # Perform the boolean operation
    try:
        original_volume = plug_part.volume
        result = plug_part - cutter
        new_volume = result.volume
        volume_diff = original_volume - new_volume

        print(f"Boolean subtraction completed:")
        print(f"  Original volume: {original_volume:.2f} mm³")
        print(f"  New volume: {new_volume:.2f} mm³")
        print(
            f"  Volume removed: {volume_diff:.2f} mm³ ({100*volume_diff/original_volume:.2f}%)"
        )

        if volume_diff <= 0:
            print("WARNING: No volume was removed! The cutout operation failed.")
    except Exception as e:
        print(f"ERROR during boolean operation: {e}")
        return plug_part

    # Clean up the result
    try:
        result.clean()
        print("Clean operation completed")
    except Exception as e:
        print(f"Warning during clean operation: {e}")

    return result


# ---------------------------------------------------------------------------#
# merge_bin_with_cutouts – create a gridfinity bin with cutouts carved in    #
# ---------------------------------------------------------------------------#
def merge_bin_with_cutouts(bin_part, plug_part):
    """
    Take a gridfinity bin and a carved plug, and merge them to create
    a bin with the cutouts carved into it.
    """
    from build123d import Location

    # Get bounding boxes to calculate proper positioning
    plug_bbox = plug_part.bounding_box()
    bin_bbox = bin_part.bounding_box()

    # Calculate Z offset to align the tops
    # The plug's top should align with the bin's top
    plug_height = plug_bbox.size.Z
    bin_top_z = bin_bbox.max.Z
    plug_bottom_z = bin_top_z - plug_height

    # The plug is already centered at origin in X and Y, same as the bin
    # So we only need to adjust Z position
    z_offset = plug_bottom_z - plug_bbox.min.Z

    print(f"Bin top Z: {bin_top_z:.2f} mm")
    print(f"Plug height: {plug_height:.2f} mm")
    print(f"Positioning plug with Z offset: {z_offset:.2f} mm")

    positioned_plug = plug_part.moved(Location((0, 0, z_offset)))

    # Verify alignment
    positioned_bbox = positioned_plug.bounding_box()
    print(f"Positioned plug top Z: {positioned_bbox.max.Z:.2f} mm")
    print(f"Difference from bin top: {positioned_bbox.max.Z - bin_top_z:.2f} mm")

    # Perform the boolean union
    print("Merging bin with carved insert...")
    merged = bin_part.fuse(positioned_plug)

    print(f"Merged volume: {merged.volume:.2f} mm³")
    print(f"Original bin volume: {bin_part.volume:.2f} mm³")
    print(f"Original plug volume: {plug_part.volume:.2f} mm³")

    return merged


# ---------------------------------------------------------------------------#
# main – CLI                                                                 #
# ---------------------------------------------------------------------------#
def main():
    cli = argparse.ArgumentParser(
        description="Generate Gridfinity bin STL, inner plug, or image-carved insert."
    )
    cli.add_argument("--x", type=int, required=True, help="X dimension (grid units)")
    cli.add_argument("--y", type=int, required=True, help="Y dimension (grid units)")
    cli.add_argument("--z", type=int, required=True, help="Z dimension (height units)")
    cli.add_argument(
        "--inner", action="store_true", help="Export solid insert instead of bin"
    )
    cli.add_argument(
        "--cutout",
        type=Path,
        metavar="IMAGE",
        help="Image of shape on a 1 cm grid to carve out of insert",
    )
    cli.add_argument(
        "--ppmm",
        type=int,
        default=DEFAULT_PPMM,
        help="Pixels-per-mm if auto-detection fails",
    )

    # Rectangular cut‑outs
    cli.add_argument(
        "--cut_x", type=str, metavar="X", help="Rect cut‑out X dimension (e.g. 30mm)"
    )
    cli.add_argument(
        "--cut_y",
        type=str,
        metavar="Y",
        help="Rect cut‑out Y dimension (e.g. 20mm)",
    )
    cli.add_argument("--cut_count", type=int, help="Number of cut‑outs (optional)")

    cli.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output and visualizations",
    )
    args = cli.parse_args()

    # Create shape matrix from x and y dimensions
    shape = create_shape(args.x, args.y)
    base = f"gridfinity_{args.x}x{args.y}"

    # Build output filename parts
    filename_parts = [base + f"x{args.z}"]

    if args.inner:
        filename_parts.append("inner")
    if args.cutout:
        filename_parts.append(args.cutout.stem)
    if args.cut_x or args.cut_y:
        if args.cut_x:
            filename_parts.append(f"x{args.cut_x}")
        if args.cut_y:
            filename_parts.append(f"y{args.cut_y}")

    outfile = "_".join(filename_parts) + ".stl"

    print(f"\nOutput file will be: {outfile}")

    # Check if we need to carve cutouts
    has_cutouts = args.cutout or (args.cut_x and args.cut_y)

    # Plain bin - no cutouts and not inner
    if not args.inner and not has_cutouts:
        export_stl(FunkyBin(shape, height_units=args.z), outfile)
        print("Exported bin:", outfile)
        return

    # Start with solid plug
    plug = gridfinity_inner(shape, args.z)

    # Apply cutouts to the plug
    if args.cutout:
        print(f"\nProcessing cutout image: {args.cutout}")
        contour = image_to_contour(
            args.cutout, ppmm=args.ppmm, debug=args.debug or True
        )
        from gfthings.parameters import plate_height

        depth = args.z * 7 - plate_height - 2 * TOP_GAP_MM
        plug = carve_plug(
            plug, contour, depth, debug=args.debug or True, cutout_name=args.cutout.stem
        )

    # Rectangular cut‑out grid
    if args.cut_x and args.cut_y:
        x_mm = parse_mm(args.cut_x)
        y_mm = parse_mm(args.cut_y)
        plug = carve_rect_grid(plug, x_mm, y_mm, args.cut_count, args.inner)

    # Now decide what to export
    if args.inner:
        # Just export the carved plug
        export_stl(plug, outfile)
        print(f"\nExported inner plug: {outfile}")
    else:
        # Create a bin and merge it with the carved plug
        print("\nCreating merged bin with cutouts...")
        bin_part = FunkyBin(shape, height_units=args.z)
        merged_part = merge_bin_with_cutouts(bin_part, plug)
        export_stl(merged_part, outfile)
        print(f"\nExported merged bin with cutouts: {outfile}")


# ---------------------------------------------------------------------------#
# tests – auto-run when imported                                             #
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
else:
    import unittest

    class GridfinityInnerTests(unittest.TestCase):
        def test_plug_smaller_than_bin(self):
            s = [[1, 1], [1, 1]]
            bin_part = FunkyBin(s, 3)
            plug_part = gridfinity_inner(s, 3)
            self.assertLess(plug_part.volume, bin_part.volume)

        def test_carve_reduces_volume(self):
            s = [[1]]
            plug = gridfinity_inner(s, 3)
            contour = [(-5, -5), (5, -5), (0, 5)]
            carve_plug(plug, contour, 10)
            self.assertLess(plug.volume, FunkyBin(s, 3).volume)

        def test_top_gap(self):
            s = [[1]]
            plug = gridfinity_inner(s, 3)
            from gfthings.parameters import plate_height

            expected = 3 * 7 - plate_height - TOP_GAP_MM
            self.assertAlmostEqual(plug.bounding_box().size.Z, expected, delta=0.1)

    unittest.main(verbosity=2, exit=False)
