import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.filters import threshold_mean
from PyQt5 import QtWidgets, QtGui, QtCore
from MIPtools import Ui_MIPtools

class MIPtoolsApp(QtWidgets.QTabWidget, Ui_MIPtools):
    def __init__(self, parent=None):
        super(MIPtoolsApp, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(lambda: self.browse_file(self.graphicsView))
        self.pushButton_7.clicked.connect(lambda: self.browse_file(self.graphicsView_3))
        self.pushButton_14.clicked.connect(lambda: self.browse_file(self.graphicsView_5))
        self.pushButton_20.clicked.connect(lambda: self.browse_file(self.graphicsView_7))
        self.pushButton_22.clicked.connect(lambda: self.browse_file(self.graphicsView_9))
        self.pushButton_24.clicked.connect(lambda: self.browse_file(self.graphicsView_11))
        self.pushButton_2.clicked.connect(lambda: self.save_image(self.graphicsView_2))
        self.pushButton_8.clicked.connect(lambda: self.save_image(self.graphicsView_4))
        self.pushButton_15.clicked.connect(lambda: self.save_image(self.graphicsView_6))
        self.pushButton_21.clicked.connect(lambda: self.save_image(self.graphicsView_8))
        self.pushButton_23.clicked.connect(lambda: self.save_image(self.graphicsView_10))
        self.pushButton_25.clicked.connect(lambda: self.save_image(self.graphicsView_12))
        self.pushButton_3.clicked.connect(self.threshold_otsu)
        self.pushButton_4.clicked.connect(self.histogram_analysis)
        self.pushButton_5.clicked.connect(self.threshold_manual)
        self.pushButton_6.clicked.connect(self.threshold_entropy)
        self.pushButton_9.clicked.connect(self.roberts)
        self.pushButton_10.clicked.connect(self.prewitt)
        self.pushButton_11.clicked.connect(self.sobel)
        self.pushButton_12.clicked.connect(self.gaussian)
        self.pushButton_13.clicked.connect(self.median)
        self.pushButton_16.clicked.connect(self.g_dilation)
        self.pushButton_17.clicked.connect(self.g_erosion)
        self.pushButton_18.clicked.connect(self.g_opening)
        self.pushButton_19.clicked.connect(self.g_closing)
        self.pushButton_26.clicked.connect(self.morphological_edge_detection)
        self.pushButton_27.clicked.connect(self.conditional_dilation)
        self.pushButton_28.clicked.connect(self.grayscale_reconstruction)
        self.pushButton_29.clicked.connect(self.morphological_gradient)
        self.pushButton_30.clicked.connect(self.b_dilation)
        self.pushButton_31.clicked.connect(self.b_erosion)
        self.pushButton_32.clicked.connect(self.b_opening)
        self.pushButton_33.clicked.connect(self.b_closing)
        self.pushButton_34.clicked.connect(self.morphological_distance_transform)
        self.pushButton_35.clicked.connect(self.morphological_skeleton)
        self.pushButton_36.clicked.connect(self.morphological_skeleton_restoration)

        self.image = None

    def browse_file(self, graphics_view):
        # Open file dialog to select image
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                            "Images (*.png *.xpm *.jpg *.bmp *.gif);;All Files (*)",
                                                            options=options)
        if file_path:
            # Load the image using QPixmap
            pixmap = QtGui.QPixmap(file_path)
            if not pixmap.isNull():
                # Display the image in the specified graphics view
                scene = QtWidgets.QGraphicsScene(self)
                scene.addPixmap(pixmap)
                graphics_view.setScene(scene)
                graphics_view.fitInView(scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

                
                # Load the image using OpenCV
                self.image = cv2.imread(file_path)
                if self.image is not None:
                    print("Image loaded successfully with OpenCV.")
                else:
                    print("Failed to load the image with OpenCV.")
            else:
                print("Failed to load the image with QPixmap.")

    def save_image(self, graphics_view):
        print("Save Image button clicked")
        scene = graphics_view.scene()
        if scene is None or not scene.items():
            print("No image to save.")
            return

        # Get the QPixmap from the QGraphicsScene
        pixmap = scene.items()[0].pixmap()

        # Convert QPixmap to QImage
        q_image = pixmap.toImage()

        # Open file dialog to select save location
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image File", "",
                                                            "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp);;All Files (*)",
                                                            options=options)
        if file_path:
            # Save QImage to file
            q_image.save(file_path)
            print(f"Image saved to {file_path}")
        else:
            print("Save operation cancelled.")


    def histogram_analysis(self):
        print("Histogram analysis button clicked")
        if self.image is None:
            print("No image loaded for histogram analysis.")
            return

        # Ensure the image is a valid numpy array
        if not isinstance(self.image, np.ndarray):
            print("Invalid image format for histogram analysis.")
            return

        # Convert image to grayscale if it is not already
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image

        # Calculate the histogram
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        # Plot the histogram using matplotlib
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel value")
        plt.ylabel("Frequency")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()

    def threshold_otsu(self):
        print("Threshold Otsu button clicked")
        if self.image is None:
            print("No image loaded for Otsu thresholding.")
            return

        # Convert image to grayscale if it is not already
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image

        # Apply Otsu's thresholding
        _, otsu_thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert the result to QPixmap and display in the graphicsView_2
        self.display_image(otsu_thresh_image, self.graphicsView_2)

    def threshold_entropy(self):
        print("Threshold Entropy button clicked")
        if self.image is None:
            print("No image loaded for entropy thresholding.")
            return

        # Convert image to grayscale if it is not already
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image

        # Calculate histogram
        grayHist = self.calcGrayHist(gray_image)
        rows, cols = gray_image.shape
        normgrayHist = grayHist / float(rows * cols)
        zeroCumuMoment = np.zeros([256], np.float32)
        for k in range(256):
            if k == 0:
                zeroCumuMoment[k] = normgrayHist[k]
            else:
                zeroCumuMoment[k] = zeroCumuMoment[k - 1] + normgrayHist[k]
        entropy = np.zeros([256], np.float32)
        for k in range(256):
            if k == 0:
                if normgrayHist[k] == 0:
                    entropy[k] = 0
                else:
                    entropy[k] = -normgrayHist[k] * np.log10(normgrayHist[k])
            else:
                if normgrayHist[k] == 0:
                    entropy[k] = entropy[k - 1]
                else:
                    entropy[k] = entropy[k - 1] - normgrayHist[k] * np.log10(normgrayHist[k])
        ft = np.zeros([256], np.float32)
        ft1, ft2 = 0., 0.
        totalEntropy = entropy[255]
        for k in range(255):
            maxfornt = np.max(normgrayHist[:k + 1])
            maxback = np.max(normgrayHist[k + 1:256])
            if (maxfornt == 0 or zeroCumuMoment[k] == 0 or maxfornt == 1 or zeroCumuMoment[k] == 1 or totalEntropy == 0):
                ft1 = 0
            else:
                ft1 = entropy[k] / totalEntropy * (np.log10(zeroCumuMoment[k]) / np.log10(maxfornt))
            if (maxback == 0 or 1 - zeroCumuMoment[k] == 0 or maxback == 1 or 1 - zeroCumuMoment[k] == 1):
                ft2 = 0
            else:
                if totalEntropy == 0:
                    ft2 = (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
                else:
                    ft2 = (1 - entropy[k] / totalEntropy) * (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
            ft[k] = ft1 + ft2
        thresloc = np.where(ft == np.max(ft))
        thresh = thresloc[0][0]
        threshold = np.copy(gray_image)
        threshold[threshold > thresh] = 255
        threshold[threshold <= thresh] = 0
        
        # Convert the result to QPixmap and display in the graphicsView_2
        self.display_image(threshold, self.graphicsView_2)

    def calcGrayHist(self, image):
        rows, cols = image.shape
        grayHist = np.zeros([256])
        for r in range(rows):
            for c in range(cols):
                grayHist[int(image[r, c])] += 1
        return grayHist
    
    def threshold_manual(self):
        print("Threshold Manual button clicked")
        if self.image is None:
            print("No image loaded for manual thresholding.")
            return

        # Convert image to grayscale if it is not already
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image

        # Prompt user for threshold value
        threshold_value, ok = QtWidgets.QInputDialog.getInt(self, "Manual Threshold", "Enter threshold value (0-255):", min=0, max=255)

        if ok:
            # Apply manual thresholding
            _, manual_thresh_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

            # Convert the result to QPixmap and display in the graphicsView_2
            self.display_image(manual_thresh_image, self.graphicsView_2)

        
    def roberts(self):
        print("Roberts operator applied")
        if self.image is None:
            print("No image loaded for Roberts operator.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        roberts_x = np.array([[1, 0], [0, -1]], dtype=int)
        roberts_y = np.array([[0, 1], [-1, 0]], dtype=int)
        roberts_x_img = cv2.filter2D(gray_image, -1, roberts_x)
        roberts_y_img = cv2.filter2D(gray_image, -1, roberts_y)
        roberts_img = np.sqrt(roberts_x_img**2 + roberts_y_img**2)
        roberts_img = (roberts_img / np.max(roberts_img) * 255).astype(np.uint8)
        self.display_image(roberts_img, self.graphicsView_4)

    def prewitt(self):
        print("Prewitt operator applied")
        if self.image is None:
            print("No image loaded for Prewitt operator.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
        prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        prewitt_x_img = cv2.filter2D(gray_image, -1, prewitt_x)
        prewitt_y_img = cv2.filter2D(gray_image, -1, prewitt_y)
        prewitt_img = np.sqrt(prewitt_x_img**2 + prewitt_y_img**2)
        prewitt_img = (prewitt_img / np.max(prewitt_img) * 255).astype(np.uint8)
        self.display_image(prewitt_img, self.graphicsView_4)

    def sobel(self):
        print("Sobel operator applied")
        if self.image is None:
            print("No image loaded for Sobel operator.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_img = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_img = (sobel_img / np.max(sobel_img) * 255).astype(np.uint8)
        self.display_image(sobel_img, self.graphicsView_4)

    def gaussian(self):
        print("Gaussian filter applied")
        if self.image is None:
            print("No image loaded for Gaussian filter.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        gaussian_img = cv2.GaussianBlur(gray_image, (5, 5), 0)
        self.display_image(gaussian_img, self.graphicsView_4)

    def median(self):
        print("Median filter applied")
        if self.image is None:
            print("No image loaded for Median filter.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        median_img = cv2.medianBlur(gray_image, 5)
        self.display_image(median_img, self.graphicsView_4)

    def g_dilation(self):
        print("Dilation button clicked")
        if self.image is None:
            print("No image loaded for dilation.")
            return

        iterations, ok = QtWidgets.QInputDialog.getInt(self, "Dilation", "Enter the number of iterations:", 1, 1, 100, 1)
        if ok:
            gray_image = self.convert_to_grayscale(self.image)
            kernel = np.ones((3, 3), np.uint8)
            dilation_img = cv2.dilate(gray_image, kernel, iterations=iterations)
            self.display_image(dilation_img, self.graphicsView_6)

    def g_erosion(self):
        print("Erosion button clicked")
        if self.image is None:
            print("No image loaded for erosion.")
            return

        iterations, ok = QtWidgets.QInputDialog.getInt(self, "Erosion", "Enter the number of iterations:", 1, 1, 100, 1)
        if ok:
            gray_image = self.convert_to_grayscale(self.image)
            kernel = np.ones((3, 3), np.uint8)
            erosion_img = cv2.erode(gray_image, kernel, iterations=iterations)
            self.display_image(erosion_img, self.graphicsView_6)

    def g_opening(self):
        print("Opening button clicked")
        if self.image is None:
            print("No image loaded for opening.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        kernel = np.ones((3, 3), np.uint8)
        opening_img = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
        self.display_image(opening_img, self.graphicsView_6)

    def g_closing(self):
        print("Closing button clicked")
        if self.image is None:
            print("No image loaded for closing.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        kernel = np.ones((3, 3), np.uint8)
        closing_img = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        self.display_image(closing_img, self.graphicsView_6)


    def morphological_edge_detection(self):
        print("Morphological Edge Detection applied")
        if self.image is None:
            print("No image loaded for edge detection.")
            return

        # Create a dialog to select edge detection method
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Edge Detection Method")
        layout = QtWidgets.QVBoxLayout()

        btn_standard = QtWidgets.QPushButton("Standard")
        btn_external = QtWidgets.QPushButton("External")
        btn_internal = QtWidgets.QPushButton("Internal")

        layout.addWidget(btn_standard)
        layout.addWidget(btn_external)
        layout.addWidget(btn_internal)

        dialog.setLayout(layout)

        btn_standard.clicked.connect(lambda: self.apply_morphological_edge_detection("standard", dialog))
        btn_external.clicked.connect(lambda: self.apply_morphological_edge_detection("external", dialog))
        btn_internal.clicked.connect(lambda: self.apply_morphological_edge_detection("internal", dialog))

        dialog.exec_()

    def apply_morphological_edge_detection(self, method, dialog):
        gray_image = self.convert_to_grayscale(self.image)
        kernel = np.ones((3, 3), np.uint8)

        if method == "standard":
            dilated = cv2.dilate(gray_image, kernel)
            eroded = cv2.erode(gray_image, kernel)
            edge_image = cv2.subtract(dilated, eroded)
        elif method == "external":
            dilated = cv2.dilate(gray_image, kernel)
            edge_image = cv2.subtract(dilated, gray_image)
        elif method == "internal":
            eroded = cv2.erode(gray_image, kernel)
            edge_image = cv2.subtract(gray_image, eroded)

        self.display_image(edge_image, self.graphicsView_8)
        dialog.accept()


    def conditional_dilation(self):
        print("Conditional Dilation applied")
        if self.image is None:
            print("No image loaded for conditional dilation.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)

        # Initial marker (e.g., center of the image)
        marker = np.zeros_like(binary_image)
        center_x, center_y = marker.shape[1] // 2, marker.shape[0] // 2
        marker[center_y-10:center_y+10, center_x-10:center_x+10] = binary_image[center_y-10:center_y+10, center_x-10:center_x+10]

        while True:
            marker_new = cv2.dilate(marker, kernel)
            marker_new = cv2.min(marker_new, binary_image)
            if np.array_equal(marker, marker_new):
                break
            marker = marker_new

        self.display_image(marker, self.graphicsView_8)

    def grayscale_reconstruction(self):
        print("Grayscale Reconstruction applied")
        if self.image is None:
            print("No image loaded for reconstruction.")
            return

        # Create a dialog to select reconstruction method
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Reconstruction Method")
        layout = QtWidgets.QVBoxLayout()

        btn_obr = QtWidgets.QPushButton("Opening by Reconstruction (OBR)")
        btn_cbr = QtWidgets.QPushButton("Closing by Reconstruction (CBR)")

        layout.addWidget(btn_obr)
        layout.addWidget(btn_cbr)

        dialog.setLayout(layout)

        btn_obr.clicked.connect(lambda: self.apply_grayscale_reconstruction("obr", dialog))
        btn_cbr.clicked.connect(lambda: self.apply_grayscale_reconstruction("cbr", dialog))

        dialog.exec_()

    def apply_grayscale_reconstruction(self, method, dialog):
        gray_image = self.convert_to_grayscale(self.image)
        kernel = np.ones((3, 3), np.uint8)

        if method == "obr":
            eroded = cv2.erode(gray_image, kernel)
            reconstructed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
        elif method == "cbr":
            dilated = cv2.dilate(gray_image, kernel)
            reconstructed = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)

        self.display_image(reconstructed, self.graphicsView_8)
        dialog.accept()

    def morphological_gradient(self):
        print("Morphological Gradient applied")
        if self.image is None:
            print("No image loaded for gradient detection.")
            return

        # Create a dialog to select gradient detection method
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Gradient Detection Method")
        layout = QtWidgets.QVBoxLayout()

        btn_standard = QtWidgets.QPushButton("Standard")
        btn_external = QtWidgets.QPushButton("External")
        btn_internal = QtWidgets.QPushButton("Internal")

        layout.addWidget(btn_standard)
        layout.addWidget(btn_external)
        layout.addWidget(btn_internal)

        dialog.setLayout(layout)

        btn_standard.clicked.connect(lambda: self.apply_morphological_gradient("standard", dialog))
        btn_external.clicked.connect(lambda: self.apply_morphological_gradient("external", dialog))
        btn_internal.clicked.connect(lambda: self.apply_morphological_gradient("internal", dialog))

        dialog.exec_()

    def apply_morphological_gradient(self, method, dialog):
        gray_image = self.convert_to_grayscale(self.image)
        kernel = np.ones((3, 3), np.uint8)

        if method == "standard":
            dilated = cv2.dilate(gray_image, kernel)
            eroded = cv2.erode(gray_image, kernel)
            gradient_image = cv2.subtract(dilated, eroded) / 2
        elif method == "external":
            dilated = cv2.dilate(gray_image, kernel)
            gradient_image = cv2.subtract(dilated, gray_image) / 2
        elif method == "internal":
            eroded = cv2.erode(gray_image, kernel)
            gradient_image = cv2.subtract(gray_image, eroded) / 2

        gradient_image = gradient_image.astype(np.uint8)
        self.display_image(gradient_image, self.graphicsView_8)
        dialog.accept()

    
    def b_dilation(self):
        print("Dilation button clicked")
        if self.image is None:
            print("No image loaded for dilation.")
            return

        iterations, ok = QtWidgets.QInputDialog.getInt(self, "Dilation", "Enter the number of iterations:", 1, 1, 100, 1)
        if ok:
            gray_image = self.convert_to_grayscale(self.image)
            kernel = np.ones((3, 3), np.uint8)
            dilation_img = cv2.dilate(gray_image, kernel, iterations=iterations)
            self.display_image(dilation_img, self.graphicsView_10)

    def b_erosion(self):
        print("Erosion button clicked")
        if self.image is None:
            print("No image loaded for erosion.")
            return

        iterations, ok = QtWidgets.QInputDialog.getInt(self, "Erosion", "Enter the number of iterations:", 1, 1, 100, 1)
        if ok:
            gray_image = self.convert_to_grayscale(self.image)
            kernel = np.ones((3, 3), np.uint8)
            erosion_img = cv2.erode(gray_image, kernel, iterations=iterations)
            self.display_image(erosion_img, self.graphicsView_10)

    def b_opening(self):
        print("Opening button clicked")
        if self.image is None:
            print("No image loaded for opening.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        kernel = np.ones((3, 3), np.uint8)
        opening_img = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
        self.display_image(opening_img, self.graphicsView_10)

    def b_closing(self):
        print("Closing button clicked")
        if self.image is None:
            print("No image loaded for closing.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        kernel = np.ones((3, 3), np.uint8)
        closing_img = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        self.display_image(closing_img, self.graphicsView_10)

    def morphological_distance_transform(self):
        print("Morphological Distance Transform applied")
        if self.image is None:
            print("No image loaded for distance transform.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        dist_transform = dist_transform.astype(np.uint8)
        self.display_image(dist_transform, self.graphicsView_12)

    def morphological_skeleton(self):
        print("Morphological Skeleton applied")
        if self.image is None:
            print("No image loaded for skeletonization.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Apply thinning to get the skeleton
        skeleton = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        self.display_image(skeleton, self.graphicsView_12)


    def morphological_skeleton_restoration(self):
        print("Morphological Skeleton Restoration applied")
        if self.image is None:
            print("No image loaded for skeleton restoration.")
            return

        gray_image = self.convert_to_grayscale(self.image)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        skeleton = np.zeros_like(binary_image)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(binary_image, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(binary_image, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary_image = eroded.copy()
            if cv2.countNonZero(binary_image) == 0:
                break
        
        # Skeleton restoration
        restored = cv2.morphologyEx(skeleton, cv2.MORPH_DILATE, kernel)
        self.display_image(restored, self.graphicsView_12)


    def convert_to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


    def display_image(self, image, graphics_view):
        """Helper function to display an image in a QGraphicsView."""
        height, width = image.shape
        bytes_per_line = width
        q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(q_image)

        scene = QtWidgets.QGraphicsScene(self)
        scene.addPixmap(pixmap)
        graphics_view.setScene(scene)
        graphics_view.fitInView(scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
   





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MIPtoolsApp()
    mainWindow.show()
    sys.exit(app.exec_())
