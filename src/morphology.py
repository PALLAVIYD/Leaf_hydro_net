import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================================================================
# Module 3: Advanced Morphological Phenotyping (OpenCV)
# Calculates angle, width reduction, curvature, and symmetry
# Ref: Section 6.4 of the Report
# ===================================================================

class BoeaMorphologyAnalyzer:
    def __init__(self, baseline_width=None):
        """
        baseline_width: W_0 represents the fully expanded leaf width in pixels.
        If unknown, it must be calibrated by measuring a healthy leaf.
        """
        self.baseline_width = baseline_width
    
    def extract_leaf_contour(self, image_path):
        """Extracts the largest green contour (the leaf) using HSV thresholding."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at {image_path}")
        
        # Convert to HSV (better for color isolation)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define Green bounds (tuned for plant leaves)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, mask, img
            
        largest = max(contours, key=cv2.contourArea)
        return largest, mask, img

    def calculate_folding_angle(self, contour):
        """
        calculates Folding Angle (theta)
        Approximates it by fitting an ellipse to the leaf and getting its orientation.
        A perfectly flat leaf perpendicular to camera = 90 deg. 
        As it folds/curls, its visual orientation changes. 
        """
        if len(contour) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            # Normalize angle to represent folding from a flat 90-degree state
            theta = 90.0 - abs(angle)
            return abs(theta)
        return 90.0
        
    def calculate_width_reduction(self, contour):
        """
        calculates Width Reduction Ratio (W / W_0)
        """
        x, y, w, h = cv2.boundingRect(contour)
        if self.baseline_width is None:
            print("Warning: Baseline width not set. Using current as baseline.")
            self.baseline_width = w
            
        ratio = w / self.baseline_width
        return ratio, w
        
    def calculate_curvature(self, contour):
        """
        calculates Curvature (kappa)
        Approximated by evaluating the convex hull area vs contour area (Solidity).
        As a leaf folds inward, it loses solidity and becomes 'curved'.
        """
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return 0.0
            
        solidity = float(area) / hull_area
        # Lower solidity = higher curvature (folding inwards)
        curvature = 1.0 - solidity 
        return curvature

    def calculate_symmetry_deviation(self, contour, filter_mask):
        """
        calculates Symmetry Deviation (sigma_sym)
        Finds the centroid, splits the mask vertically at the centroid,
        and calculates the absolute difference in area between Left and Right halves.
        """
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0
            
        cX = int(M["m10"] / M["m00"])
        
        h, w = filter_mask.shape
        left_half = filter_mask[:, :cX]
        right_half = filter_mask[:, cX:]
        
        left_area = cv2.countNonZero(left_half)
        right_area = cv2.countNonZero(right_half)
        
        total_area = left_area + right_area
        if total_area == 0:
            return 0.0
            
        # Symmetry index = difference over total
        symmetry = abs(left_area - right_area) / total_area
        return symmetry

    def analyze_image(self, image_path):
        """Runs the full pipeline on a single image."""
        contour, mask, original_img = self.extract_leaf_contour(image_path)
        
        if contour is None:
            return {"Error": "No leaf found."}
            
        theta = self.calculate_folding_angle(contour)
        w_ratio, current_width = self.calculate_width_reduction(contour)
        kappa = self.calculate_curvature(contour)
        sigma_sym = self.calculate_symmetry_deviation(contour, mask)
        
        results = {
            "Folding_Angle_Theta": round(theta, 2),
            "Width_Ratio_w": round(w_ratio, 3),
            "Current_Width": current_width,
            "Curvature_Kappa": round(kappa, 3),
            "Symmetry_Deviation_Sigma": round(sigma_sym, 3)
        }
        return results

if __name__ == "__main__":
    print("--- BoeaMorphologyAnalyzer ---")
    analyzer = BoeaMorphologyAnalyzer(baseline_width=300) # Example baseline
    
    # You can test it by passing an image:
    # results = analyzer.analyze_image('test_leaf.jpg')
    # print(results)
    print("Analyzer ready to ingest images.")
