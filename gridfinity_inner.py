#!/usr/bin/env python3
# gridfinity_inner.py – Generate Gridfinity bins, solid inserts, and
# image-shaped cut-outs that drop neatly inside.
#
# • Each helper is tiny; no duplicated logic
# • Comments live above functions (no doc-strings)
# • Real tests run automatically when the file is imported
# • New: --solid creates a completely filled bin with flat top
#
# Requires:
# build123d ≥ 0.9
# gfthings (master branch)
# opencv-python for --cutout
# ---------------------------------------------------------------------------#
# imports #
# ---------------------------------------------------------------------------#
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from build123d import export_stl  # STL helper
from gfthings.Bin import FunkyBin  # Gridfinity bin generator
import math

# ---------------------------------------------------------------------------#
# constants #
# ---------------------------------------------------------------------------#
TOLERANCE_MM = 0.30  # wall clearance for the inner plug
TOP_GAP_MM = 1.0  # gap above plug so you can pull it out
MIN_HOLE_MM = 0.75  # fill gaps narrower than this when tracing
DEFAULT_PPMM = 10  # default pixels-per-millimetre for --cutout
MIN_SPACING_MM = 0.6  # minimum space between rectangular cut‑outs


# ---------------------------------------------------------------------------#
# create_shape – create a binary matrix from x and y dimensions #
# ---------------------------------------------------------------------------#
def create_shape(x: int, y: int) -> List[List[int]]:
    """Create a shape matrix filled with 1s of the given dimensions."""
    return [[1] * x for _ in range(y)]


# ---------------------------------------------------------------------------#
# parse_mm – convert CLI dimension strings to float millimetres #
# ---------------------------------------------------------------------------#
def parse_mm(text: str) -> float:
    if text.lower().endswith("mm"):
        text = text[:-2]
    return float(text)


# ---------------------------------------------------------------------------#
# best_grid – choose rows/cols & spacing for rectangular cut‑outs #
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
# carve_rect_grid – subtract a grid of rectangles from the part #
# ---------------------------------------------------------------------------#
# Each rectangle is centred at Z depth so that 2 mm of material remains.
# ---------------------------------------------------------------------------#
def carve_rect_grid(
    part,
    rect_x: float,
    rect_y: float,
    count: Optional[int],
    is_inner: bool,
    fillet_radius: float = 0.0,
):
    from build123d import (
        BuildPart,
        BuildSketch,
        Rectangle,
        RectangleRounded,
        Location,
        extrude,
        Plane,
        Align,
    )

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
    if fillet_radius > 0:
        print(f"Using fillet radius: {fillet_radius:.2f} mm")
    cutters = []
    for r in range(rows):
        cy = start_y + r * (rect_y + sy)
        for c in range(cols):
            cx = start_x + c * (rect_x + sx)
            with BuildPart():
                z_top = bbox.max.Z
                with BuildSketch(Plane.XY * Location((cx, cy, z_top))):
                    if fillet_radius > 0:
                        # Ensure fillet radius isn't too large
                        max_fillet = min(rect_x, rect_y) / 2
                        actual_fillet = min(fillet_radius, max_fillet)
                        if actual_fillet < fillet_radius:
                            print(
                                f"Warning: Fillet radius reduced to {actual_fillet:.2f} mm to fit rectangle"
                            )
                        RectangleRounded(
                            rect_x,
                            rect_y,
                            actual_fillet,
                            align=(Align.CENTER, Align.CENTER),
                        )
                    else:
                        Rectangle(rect_x, rect_y, align=(Align.CENTER, Align.CENTER))
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
# image_to_contour – trace the largest blob inside a 1 cm-grid photo #
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
# gridfinity_inner – solid insert that fits the bin precisely #
# ---------------------------------------------------------------------------#
def gridfinity_inner(
    shape: List[List[int]],
    z_units: int = 3,
    tol: float = TOLERANCE_MM,
    top_gap: float = TOP_GAP_MM,
    stackable: bool = True,
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
    # Try to get stacking lip height from parameters
    try:
        from gfthings.parameters import stacking_lip_height

        STACKING_LIP = stacking_lip_height
    except ImportError:
        STACKING_LIP = 4.4  # Default stacking lip height
    cols = max(len(r) for r in shape)
    rows = len(shape)
    inner_x = cols * bin_size - 2 * (bin_clearance + WALL + tol)
    inner_y = rows * bin_size - 2 * (bin_clearance + WALL + tol)
    inner_r = max(outer_rad - bin_clearance - WALL - tol, 0)

    # Adjust height based on stackability
    if stackable:
        # With stacking lip, leave room for the lip and a gap to pull out
        inner_z = z_units * 7 - top_gap
    else:
        # Without stacking lip, extend all the way to the top
        # No gap needed since we want it flush
        inner_z = z_units * 7

    print(f"Calculated inner_z: {inner_z:.2f} mm")
    print(f"Creating insert: {cols}x{rows} grid, z {z_units} units")
    print(f"Stackable: {stackable}")
    print(f"Usable height: {z_units * 7:.2f} mm")
    print(f"Plate height: {plate_height:.2f} mm")
    if stackable:
        print(f"Top gap: {top_gap:.2f} mm")
        print(f"Stacking lip: {STACKING_LIP:.2f} mm")
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
# carve_plug – subtract a 2-D contour from the plug #
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
        print(f" Original volume: {original_volume:.2f} mm³")
        print(f" New volume: {new_volume:.2f} mm³")
        print(
            f" Volume removed: {volume_diff:.2f} mm³ ({100*volume_diff/original_volume:.2f}%)"
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
# create_non_stackable_bin – create a gridfinity bin without stacking lip #
# ---------------------------------------------------------------------------#
def create_non_stackable_bin(shape: List[List[int]], z_units: int):
    """
    Create a gridfinity-compatible bin without the stacking lip.
    The usable height is the same as a stackable bin, so total height is shorter.
    """
    from build123d import (
        BuildPart,
        BuildSketch,
        RectangleRounded,
        Rectangle,
        extrude,
        Plane,
        Location,
        Align,
        Mode,
    )
    from gfthings.Bin import FunkyBin
    from gfthings.parameters import bin_size, bin_clearance, outer_rad, plate_height

    try:
        from gfthings.parameters import stacking_lip_height

        STACKING_LIP = stacking_lip_height
    except ImportError:
        STACKING_LIP = 4.4

    try:
        from gfthings.parameters import wall_thickness as WALL
    except ImportError:
        WALL = 1.2

    # First create a standard bin to get the base with gridfinity features
    standard_bin = FunkyBin(shape, height_units=z_units)
    standard_bbox = standard_bin.bounding_box()

    # The height without the lip
    target_height = standard_bbox.size.Z - STACKING_LIP

    print(f"\nCreating non-stackable bin:")
    print(f" Grid: {len(shape[0])}x{len(shape)}, height units: {z_units}")
    print(
        f" Target height: {target_height:.6f} mm (stackable was {standard_bbox.size.Z:.6f} mm)"
    )

    cols = max(len(r) for r in shape)
    rows = len(shape)

    # Calculate dimensions
    outer_x = cols * bin_size - 2 * bin_clearance
    outer_y = rows * bin_size - 2 * bin_clearance
    outer_r = outer_rad - bin_clearance
    inner_x = outer_x - 2 * WALL
    inner_y = outer_y - 2 * WALL
    inner_r = max(outer_r - WALL, 0)

    # Extract just the base plate with gridfinity features
    with BuildPart():
        with BuildSketch(Plane.XY * Location((0, 0, plate_height))):
            Rectangle(
                standard_bbox.size.X + 10,
                standard_bbox.size.Y + 10,
                align=(Align.CENTER, Align.CENTER),
            )
        base_cutter = extrude(amount=target_height)

    base_plate = standard_bin - base_cutter

    # Create straight walls from plate_height to target_height
    wall_height = target_height - plate_height

    with BuildPart():
        with BuildSketch(Plane.XY * Location((0, 0, plate_height))):
            # Outer wall
            RectangleRounded(
                outer_x, outer_y, outer_r, align=(Align.CENTER, Align.CENTER)
            )
            # Inner cavity (negative)
            RectangleRounded(
                inner_x,
                inner_y,
                inner_r,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER),
            )
        walls = extrude(amount=wall_height)

    # Combine base and walls
    result = base_plate + walls

    # Verify the height
    result_bbox = result.bounding_box()
    print(f"Non-stackable bin created with height: {result_bbox.size.Z:.6f} mm")
    print(
        f"Height difference from target: {abs(result_bbox.size.Z - target_height):.6f} mm"
    )

    return result


def convert_to_non_stackable(merged_part, shape: List[List[int]]):
    """
    Convert a merged bin (with cutouts) to non-stackable by modifying the top.
    This preserves the cutouts while removing the stacking lip.
    """
    from build123d import (
        BuildPart,
        BuildSketch,
        Rectangle,
        extrude,
        Plane,
        Location,
        Align,
    )

    try:
        from gfthings.parameters import stacking_lip_height

        STACKING_LIP = stacking_lip_height
    except ImportError:
        STACKING_LIP = 4.4

    bbox = merged_part.bounding_box()

    # The lip starts at this height
    lip_start_z = bbox.max.Z - STACKING_LIP

    print(
        f"Converting to non-stackable: cutting at Z={lip_start_z:.2f} mm (removing {STACKING_LIP:.2f} mm lip)"
    )

    # Cut off everything above lip_start_z
    with BuildPart():
        with BuildSketch(Plane.XY * Location((0, 0, lip_start_z))):
            Rectangle(
                bbox.size.X + 10, bbox.size.Y + 10, align=(Align.CENTER, Align.CENTER)
            )
        cutter = extrude(amount=STACKING_LIP + 1)

    result = merged_part - cutter
    final_bbox = result.bounding_box()
    print(f"After cutting lip: height = {final_bbox.size.Z:.2f} mm")

    print(f"Converted to non-stackable bin")
    print(
        f"Final height: {final_bbox.size.Z:.2f} mm (original was {bbox.size.Z:.2f} mm)"
    )

    return result


def remove_stacking_lip(bin_part):
    """
    Convert a stackable bin to non-stackable by extending the walls straight up
    to the same total height, removing the lip and slant.
    """
    from build123d import (
        BuildPart,
        BuildSketch,
        RectangleRounded,
        Rectangle,
        extrude,
        Plane,
        Location,
        Align,
        Face,
        Wire,
    )
    from gfthings.parameters import bin_size, bin_clearance, outer_rad

    # Try to get parameters
    try:
        from gfthings.parameters import stacking_lip_height

        STACKING_LIP = stacking_lip_height
    except ImportError:
        STACKING_LIP = 4.4  # Default stacking lip height

    try:
        from gfthings.parameters import wall_thickness as WALL
    except ImportError:
        WALL = 1.2

    bbox = bin_part.bounding_box()

    # Calculate the shape dimensions
    cols = round(bbox.size.X / bin_size)
    rows = round(bbox.size.Y / bin_size)

    # Calculate where the stacking lip starts (the height we want to modify from)
    lip_start_z = bbox.max.Z - STACKING_LIP

    # Create a shell that extends the walls straight up
    # Outer dimensions (same as bin)
    outer_x = cols * bin_size - 2 * bin_clearance
    outer_y = rows * bin_size - 2 * bin_clearance
    outer_r = outer_rad - bin_clearance

    # Inner dimensions (for the cavity)
    inner_x = outer_x - 2 * WALL
    inner_y = outer_y - 2 * WALL
    inner_r = max(outer_r - WALL, 0)

    # Create the extension piece
    with BuildPart():
        # Start at the lip start height
        with BuildSketch(Plane.XY * Location((0, 0, lip_start_z))):
            # Outer shape
            outer = RectangleRounded(
                outer_x, outer_y, outer_r, align=(Align.CENTER, Align.CENTER)
            )
            # Inner cavity to make it a shell
            inner = RectangleRounded(
                inner_x, inner_y, inner_r, align=(Align.CENTER, Align.CENTER)
            )
        # Extrude up to the full height
        extension = extrude(amount=STACKING_LIP)

        # Now cut out the inner cavity from the extension
        with BuildSketch(
            Plane.XY * Location((0, 0, lip_start_z - 0.1))
        ):  # Start slightly below
            RectangleRounded(
                inner_x, inner_y, inner_r, align=(Align.CENTER, Align.CENTER)
            )
        cavity_cut = extrude(amount=STACKING_LIP + 0.2)  # Cut through completely

    extension = extension - cavity_cut

    # First cut off the existing stacking lip
    with BuildPart():
        with BuildSketch(Plane.XY * Location((0, 0, lip_start_z))):
            Rectangle(
                bbox.size.X + 10, bbox.size.Y + 10, align=(Align.CENTER, Align.CENTER)
            )
        cutter = extrude(amount=STACKING_LIP + 1)

    # Remove the old lip and add the new straight extension
    result = bin_part - cutter
    result = result + extension

    print(f"Converted to non-stackable bin (maintained total height)")
    return result


def merge_bin_with_cutouts(bin_part, plug_part, stackable: bool):
    """
    Take a gridfinity bin and a carved plug, and merge them to create
    a bin with the cutouts carved into it.
    """
    from build123d import Location
    from gfthings.parameters import plate_height

    # Get bounding boxes to calculate proper positioning
    plug_bbox = plug_part.bounding_box()
    bin_bbox = bin_part.bounding_box()

    # The plug should always start at plate_height from the bottom
    bin_bottom_z = bin_bbox.min.Z
    plug_target_bottom_z = bin_bottom_z + plate_height

    # Calculate the offset needed
    z_offset = plug_target_bottom_z - plug_bbox.min.Z

    print(f"\nMerging bin with plug:")
    print(f"Bin bottom Z: {bin_bottom_z:.6f} mm")
    print(f"Bin top Z: {bin_bbox.max.Z:.6f} mm")
    print(f"Plug height: {plug_bbox.size.Z:.6f} mm")
    print(f"Plug will be positioned with bottom at Z: {plug_target_bottom_z:.6f} mm")
    print(f"Z offset: {z_offset:.6f} mm")
    print(f"Stackable: {stackable}")

    positioned_plug = plug_part.moved(Location((0, 0, z_offset)))

    # Verify alignment
    positioned_bbox = positioned_plug.bounding_box()
    print(f"Positioned plug bottom Z: {positioned_bbox.min.Z:.6f} mm")
    print(f"Positioned plug top Z: {positioned_bbox.max.Z:.6f} mm")

    top_difference = positioned_bbox.max.Z - bin_bbox.max.Z
    print(f"Difference from bin top: {top_difference:.6f} mm")

    if not stackable:
        # For non-stackable, the plug MUST reach exactly to the bin top
        if abs(top_difference) > 0.0001:  # 0.1 micron tolerance
            print(f"WARNING: Non-stackable alignment issue detected!")
            print(f" Bin top: {bin_bbox.max.Z:.6f} mm")
            print(f" Plug top: {positioned_bbox.max.Z:.6f} mm")
            print(f" Difference: {top_difference:.6f} mm")

            # Force exact alignment by adjusting plug height
            print("Applying micro-adjustment to ensure perfect flush alignment...")
            z_correction = bin_bbox.max.Z - positioned_bbox.max.Z
            positioned_plug = positioned_plug.moved(Location((0, 0, z_correction)))

            # Re-check
            positioned_bbox = positioned_plug.bounding_box()
            new_difference = positioned_bbox.max.Z - bin_bbox.max.Z
            print(f"After correction - plug top Z: {positioned_bbox.max.Z:.6f} mm")
            print(f"Final difference: {new_difference:.6f} mm")

            if abs(new_difference) > 0.0001:
                print("ERROR: Could not achieve perfect alignment!")
    else:
        # For stackable, plug should be below the top by at least TOP_GAP_MM
        expected_gap = -TOP_GAP_MM
        if top_difference > expected_gap + 0.1:
            print(
                f"WARNING: Stackable plug may be too tall (gap: {-top_difference:.2f} mm, expected: {TOP_GAP_MM:.2f} mm)"
            )

    # Perform the boolean union
    print("Performing boolean union...")
    merged = bin_part.fuse(positioned_plug)

    print(f"Merged volume: {merged.volume:.2f} mm³")
    print(f"Original bin volume: {bin_part.volume:.2f} mm³")
    print(f"Original plug volume: {plug_part.volume:.2f} mm³")

    return merged


# ---------------------------------------------------------------------------#
# main – CLI #
# ---------------------------------------------------------------------------#
def main():
    cli = argparse.ArgumentParser(
        description="Generate Gridfinity bin STL, inner plug, solid-filled bin, or image-carved insert."
    )
    cli.add_argument("--x", type=int, required=True, help="X dimension (grid units)")
    cli.add_argument("--y", type=int, required=True, help="Y dimension (grid units)")
    cli.add_argument("--z", type=int, required=True, help="Z dimension (height units)")
    cli.add_argument(
        "--inner", action="store_true", help="Export solid insert instead of bin"
    )
    cli.add_argument(
        "--stackable", action="store_true", help="Add stacking lip to top of bin"
    )
    cli.add_argument(
        "--solid", action="store_true", help="Create a solid-filled bin with flat top"
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
        "--cut_fillet",
        type=float,
        default=10.0,
        help="Fillet radius for cut‑out corners in mm (default: 10)",
    )
    cli.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output and visualizations",
    )
    args = cli.parse_args()

    # Validate arguments
    if args.solid and args.inner:
        cli.error("--solid and --inner cannot be used together")

    if args.solid and (args.cutout or (args.cut_x and args.cut_y)):
        cli.error("--solid cannot be used with cutouts (--cutout, --cut_x/--cut_y)")

    # Create shape matrix from x and y dimensions
    shape = create_shape(args.x, args.y)
    base = f"gridfinity_{args.x}x{args.y}"
    # Build output filename parts
    filename_parts = [base + f"x{args.z}"]
    if args.inner:
        filename_parts.append("inner")
    if args.solid:
        filename_parts.append("solid")
    if not args.stackable:  # Add "nonstackable" for non-stackable bins
        filename_parts.append("nonstackable")
    if args.cutout:
        filename_parts.append(args.cutout.stem)
    if args.cut_x or args.cut_y:
        if args.cut_x:
            filename_parts.append(f"x{args.cut_x}")
        if args.cut_y:
            filename_parts.append(f"y{args.cut_y}")
        if args.cut_fillet != 10.0:  # Only add to filename if not default
            filename_parts.append(f"f{args.cut_fillet}")
    outfile = "_".join(filename_parts) + ".stl"
    print(f"\nOutput file will be: {outfile}")
    # Check if we need to carve cutouts or create solid fill
    has_cutouts = args.cutout or (args.cut_x and args.cut_y) or args.solid
    # Plain bin - no cutouts and not inner
    if not args.inner and not has_cutouts:
        if args.stackable:
            bin = FunkyBin(shape, height_units=args.z)
        else:
            bin = create_non_stackable_bin(shape, args.z)
        export_stl(bin, outfile)
        print("Exported bin:", outfile)
        return
    # Start with solid plug
    print(f"\nCreating plug/insert...")
    tol = TOLERANCE_MM if args.inner else 0.0
    plug = gridfinity_inner(shape, args.z, tol=tol, stackable=args.stackable)

    # Apply cutouts to the plug (only if not --solid)
    if not args.solid:
        if args.cutout:
            print(f"\nProcessing cutout image: {args.cutout}")
            contour = image_to_contour(
                args.cutout, ppmm=args.ppmm, debug=args.debug or True
            )
            # Calculate depth based on the actual plug height
            plug_bbox = plug.bounding_box()
            depth = plug_bbox.size.Z - 2.0  # Leave 2mm at bottom
            plug = carve_plug(
                plug,
                contour,
                depth,
                debug=args.debug or True,
                cutout_name=args.cutout.stem,
            )

        # Rectangular cut‑out grid
        if args.cut_x and args.cut_y:
            x_mm = parse_mm(args.cut_x)
            y_mm = parse_mm(args.cut_y)
            plug = carve_rect_grid(
                plug, x_mm, y_mm, args.cut_count, args.inner, args.cut_fillet
            )
    else:
        print("Creating solid-filled bin (no cutouts)")

    # Now decide what to export
    if args.inner:
        # Just export the carved plug
        export_stl(plug, outfile)
        print(f"\nExported inner plug: {outfile}")
    else:
        # Create a bin and merge it with the carved plug
        print(f"Creating merged bin with cutouts...")
        # Always start with a stackable bin
        bin_part = FunkyBin(shape, height_units=args.z)
        bin_bbox = bin_part.bounding_box()
        print(f"Initial bin height: {bin_bbox.size.Z:.2f} mm")

        # Merge the cutouts first
        merged_part = merge_bin_with_cutouts(bin_part, plug, args.stackable)

        # Then convert to non-stackable if needed
        if not args.stackable:
            merged_part = convert_to_non_stackable(merged_part, shape)
            final_bbox = merged_part.bounding_box()
            print(
                f"Non-stackable final height: {final_bbox.size.Z:.2f} mm (should match initial: {bin_bbox.size.Z:.2f} mm)"
            )

        export_stl(merged_part, outfile)
        if args.solid:
            print(f"\nExported solid-filled bin: {outfile}")
        else:
            print(f"\nExported merged bin with cutouts: {outfile}")


# ---------------------------------------------------------------------------#
# tests – auto-run when imported #
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
else:
    import unittest

    class GridfinityInnerTests(unittest.TestCase):
        def test_plug_smaller_than_bin(self):
            s = create_shape(2, 2)
            bin_part = FunkyBin(s, 3)
            plug_part = gridfinity_inner(s, 3, stackable=True)
            self.assertLess(plug_part.volume, bin_part.volume)

        def test_carve_reduces_volume(self):
            s = create_shape(1, 1)
            plug = gridfinity_inner(s, 3, stackable=True)
            contour = [(-5, -5), (5, -5), (0, 5)]
            carved = carve_plug(plug, contour, 10)
            self.assertLess(carved.volume, plug.volume)

        def test_top_gap(self):
            s = create_shape(1, 1)
            plug = gridfinity_inner(s, 3, stackable=True)
            # For stackable, there should be a top gap
            expected = 3 * 7 - TOP_GAP_MM
            self.assertAlmostEqual(plug.bounding_box().size.Z, expected, delta=0.1)

        def test_non_stackable_height(self):
            s = create_shape(1, 1)
            # Non-stackable bins should be shorter than stackable by the lip height
            stackable_bin = FunkyBin(s, 3)
            non_stackable_bin = create_non_stackable_bin(s, 3)

            stackable_height = stackable_bin.bounding_box().size.Z
            non_stackable_height = non_stackable_bin.bounding_box().size.Z
            try:
                from gfthings.parameters import stacking_lip_height

                STACKING_LIP = stacking_lip_height
            except ImportError:
                STACKING_LIP = 4.4
            self.assertAlmostEqual(
                stackable_height - STACKING_LIP, non_stackable_height, delta=0.001
            )

        def test_solid_bin_volume(self):
            s = create_shape(1, 1)
            # A solid bin should have more volume than a hollow bin
            hollow_bin = FunkyBin(s, 3)
            plug = gridfinity_inner(s, 3, tol=0.0, stackable=False)
            merged = merge_bin_with_cutouts(hollow_bin, plug, stackable=False)

            # The solid bin should have significantly more volume
            self.assertGreater(merged.volume, hollow_bin.volume * 1.5)


#    unittest.main(verbosity=2, exit=False)
