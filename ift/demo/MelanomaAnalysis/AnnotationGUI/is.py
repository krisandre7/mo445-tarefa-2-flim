#!/usr/bin/python2.7
#is.py: Interactive Segmentation

import pyift
from Tkinter import *
import tkFileDialog
import tkColorChooser
from math import sqrt
from shutil import copyfile
import os
import zipfile

class InteractiveSegmentationApp(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        master.title("Interactive Segmentation")
        
        self.color_labels = ["Label 1", "Label 2", 'boundary']
        self.color_dict = { self.color_labels[0] : (76,85,255), self.color_labels[1] : (225,1,148), 'boundary' : (41, 240, 110)}
        
        self.ift_image = None
        self.seed_image = None
        self.markers_image = None
        #Suggestion
        self.suggestion_step = "ROOTS"
        self.reducedMST = None
        self.trainGraph = None
        self.reducedZ = None
        self.graph = None
        
        self.current_folder = ""
        self.current_images = []
        self.current_image_index = 0

        #Constants
        self.STEPS = 0
        self.VOL_RATIO = 0.2
        
        self.K_MAX_PROPORTION = 0.02
        self.SAMPLES_PER_SUGGESTION = 8
        
        self.create_widgets()
        
    def create_widgets(self):
        self.pack(expand = True, fill = IFT_BOTH)

        #Menu
        self.menubar = Menu(self.master)
        
        filemenu = Menu(self.menubar, tearoff=0)
        #filemenu.add_command(label = "Open Image...", underline = 0, command = self.open_image)
        filemenu.add_command(label = "Open Image Database...", underline =5, command = self.open_image_folder)
        filemenu.add_command(label = "Export Results...", command = self.export_results)
        #filemenu.add_separator()
        #filemenu.add_command(label = "Load Markers...", command = self.load_markers)
        #filemenu.add_command(label = "Save Markers...", command = self.save_markers)
        #filemenu.add_separator()
        self.menubar.add_cascade(label = "File", menu = filemenu)
        
        save_seg_menu = Menu(self.menubar,tearoff = 0)
        save_seg_menu.add_command(label = "All superpixels...", command = self.save_all_superpixels)
        save_seg_menu.add_command(label = "By classification...", command = self.save_cl_classification_image) 
        save_seg_menu.add_command(label = "By superpixels...", command = self.save_super_classification_image)
        save_seg_menu.add_command(label = "By superpixels (mask)...", command = self.save_super_mask)            
        save_seg_menu.add_command(label = "By pixels...", command = self.save_pixel_classification_image)
        #filemenu.add_cascade(label = "Save Segmentation", menu = save_seg_menu) 
        
        self.master.config(menu=self.menubar)
        
        #Left Frame
        self.left_frame = Frame(self, relief = RAISED, borderwidth = 1)
        self.left_frame.pack(side = LEFT, anchor = NE, expand = False)
        
        self.image_frame = LabelFrame(self.left_frame, text = "Image")
        self.image_frame.pack(side = TOP,fill = X);
        self.button_redraw_image = Button(self.image_frame, text = "Redraw", underline = 2, command = self.redraw_image)
        self.button_redraw_image.pack(side = TOP)
        self.button_remove_markers = Button(self.image_frame, text = "Remove Markers", underline = 0, command = self.remove_markers)
        self.button_remove_markers.pack(side = TOP)
        
        self.marker_frame = LabelFrame(self.left_frame, text = "Marker")
        #self.marker_frame.pack(side = TOP, fill = X)
        size_label = Label(self.marker_frame, text = "Size:")
        size_label.grid(row = 0, column = 0)
        self.size_spinbox = Spinbox(self.marker_frame, from_ = 3, to = 50, width = 2)
        self.size_spinbox.delete(0,END)
        self.size_spinbox.insert(0,"10")
        self.size_spinbox.grid(row = 0, column = 1)

        color = self._convert_ycbcr_to_rgb_str(self.color_dict[self.color_labels[0]])
        self.button_color = Button(self.marker_frame, bg = color, activebackground=color, relief = RAISED, borderwidth = 1, command = self.show_label_color_dialog)
        self.button_color.grid(row=1,column = 0)
        
        self.current_label = StringVar(self.marker_frame)
        self.current_label.set(self.color_labels[1])
        self.current_label.trace("w", self.handle_label_changed)
        self.label_listbox = None
        self.redefine_label_listbox()
        self.handle_label_changed()
        
        self.button_new_label = Button(self.marker_frame, text = "+", command = self.add_label)
        self.button_new_label.grid(row=1,column = 2)
        
        self.segmentation_frame = LabelFrame(self.left_frame, text = "Segmentation")
        self.segmentation_frame.pack(side= TOP, fill = X)
        label_spatial_radius = Label(self.segmentation_frame, text = "Spatial Radius:")
        #label_spatial_radius.grid(row = 0, column = 0)
        self.entry_spatial_radius = Entry(self.segmentation_frame, width = 8)
        self.entry_spatial_radius.insert(0, "3.0")
        self.entry_spatial_radius.bind("<Return>",self.handle_superpixel_parameter_change)
        #self.entry_spatial_radius.grid(row = 0, column = 1)
        
        label_volume_threshold = Label(self.segmentation_frame, text = "Volume:")
        label_volume_threshold.grid(row = 0, column = 0)
        self.entry_volume_threshold = Entry(self.segmentation_frame, width = 8)
        self.entry_volume_threshold.insert(0, "150")
        self.entry_volume_threshold.bind("<Return>",self.handle_superpixel_parameter_change)
        self.entry_volume_threshold.grid(row = 0, column = 1)
        
        color = self._convert_ycbcr_to_rgb_str(self.color_dict['boundary'])
        self.button_color = Button(self.segmentation_frame, bg = color, activebackground=color, relief = RAISED, borderwidth = 1, command = self.show_boundary_color_dialog)
        self.button_color.grid(row=1,column = 0)
        self.button_superpixel_classify = Button(self.segmentation_frame,text = "Segment", underline = 0, command = self.show_superpixel_classification)
        self.button_superpixel_classify.grid(row = 1, column = 1)
        self.button_segmentation_save = Button(self.segmentation_frame, text = "Save", command = self.handle_button_save)
        self.button_segmentation_save.grid(row=2,column = 0, columnspan = 2)
        
        self.show_superpixels = IntVar()
        self.toggle_segmentation = Checkbutton(self.segmentation_frame, text="View", variable=self.show_superpixels, command=self.redraw_image)
        #self.toggle_segmentation.grid(row = 2, column = 0)
        self.button_ok_parameters = Button(self.segmentation_frame, text="OK",command=self.handle_superpixel_parameter_change)
        #self.button_ok_parameters.grid(row=2,column = 1)
        
        self.scaler_nsuperpixels = Scale(self.segmentation_frame, from_=0, to=1000, orient=HORIZONTAL)
        #self.scaler_nsuperpixels.grid(row=3,column = 0,columnspan=2)
        self.scaler_nsuperpixels.set(0)
        
        self.classification_frame = LabelFrame(self.left_frame, text = "Segmentation")
        #self.classification_frame.pack(side = TOP, fill = X)
        self.button_superpixel_classify = Button(self.classification_frame,text = "Superpixel", underline = 0, command = self.show_superpixel_classification)
        self.button_superpixel_classify.pack(side = TOP)
        self.button_pixel_classify = Button(self.classification_frame, text = "Pixel", underline = 0, command = self.show_pixel_classification)
        self.button_pixel_classify.pack(side = TOP)
        self.button_cl_classify = Button(self.classification_frame,text="Classification", underline=0, command = self.show_class_classification)
        self.button_cl_classify.pack(side = TOP)
        
        self.suggestion_frame = LabelFrame(self.left_frame, text = "Suggestions")
        #self.suggestion_frame.pack(side = TOP,fill = X)
        self.button_next_suggestion = Button(self.suggestion_frame,text = "Next", command = self.next_suggestion)
        self.button_next_suggestion.pack(side = TOP)

        #Right Frame
        self.right_frame = Frame(self)
        self.right_frame.pack(side = RIGHT, anchor = NE, expand = True, fill = IFT_BOTH)
        
        #Canvas Frame
        self.canvas_frame = Frame(self.right_frame, relief = SUNKEN, borderwidth = 1)
        self.canvas_frame.pack(expand = True, fill = IFT_BOTH)
        
        self.image_canvas = Canvas(self.canvas_frame,borderwidth=0,width = 640, height = 480)
        self.image_canvas.image_id = None

        self.image_hscroll = Scrollbar(self.canvas_frame, orient = HORIZONTAL)
        self.image_hscroll.pack(side=BOTTOM,fill = X)
        self.image_hscroll.config(command=self.image_canvas.xview)
        
        self.image_vscroll = Scrollbar(self.canvas_frame,orient = VERTICAL)
        self.image_vscroll.pack(side = RIGHT, fill = Y)
        self.image_vscroll.config(command=self.image_canvas.yview)

        self.image_canvas.config(yscrollcommand = self.image_vscroll.set,xscrollcommand=self.image_hscroll.set)
        
        self.image_canvas.pack(side = LEFT,expand = True, fill = IFT_BOTH)
        
        #Navigation Frame
        self.dummy_navigation_frame = Frame(self.right_frame, relief = RAISED, borderwidth = 1)
        self.dummy_navigation_frame.pack(side = BOTTOM, fill = X)
        
        self.navigation_frame = Frame(self.dummy_navigation_frame,borderwidth = 0)
        self.navigation_frame.pack()
        self.button_previous = Button(self.navigation_frame,text = "< Previous",command = self.previous_image)
        self.button_previous.pack(side = LEFT,padx = 3)
        
        self.current_image_var = IntVar(self.navigation_frame)
        self.current_image_var.set(0)
        self.entry_current_image = Entry(self.navigation_frame,width = 5,textvariable = self.current_image_var)
        self.entry_current_image.pack(side = LEFT,anchor = CENTER)
        self.entry_current_image.bind("<Return>",self.handle_current_image_change)
        
        self.label_total_images = Label(self.navigation_frame, text = "/ 0")
        self.label_total_images.pack(side = LEFT)
        self.button_next = Button(self.navigation_frame, text = "Next >", command = self.next_image)
        self.button_next.pack(side = LEFT,padx = 3)
        
        #Mouse Events
        self.image_canvas.bind("<B1-Motion>", self.handle_canvas_lclick)
        self.image_canvas.bind("<Button-1>", self.handle_canvas_lclick)
        self.image_canvas.bind("<B3-Motion>", self.handle_canvas_rclick)
        self.image_canvas.bind("<Button-3>", self.handle_canvas_rclick)

        #Other Events:
        self.master.protocol("WM_DELETE_WINDOW",self.quit)
        
    def draw_superpixels_on(self,source,destination, write_label_image = False):
        if self.scaler_nsuperpixels.get() != 0:     
            adj1 = pyift.iftCircular(float(self.entry_spatial_radius.get()))
            adj2 = pyift.iftCircular(1.0)
        
            #The 7 alphas weight: mean band1, mean band2, mean band3, relative area, color histogram intersection, lbp histogram, rectangularity
            alpha = [0.2, 1.0, 1.0, 0.0, 0.0, 0.0 ]
        
            basins = pyift.iftImageBasins(self.ift_image, adj1)
            marker = pyift.iftVolumeClose(basins, int(self.entry_volume_threshold.get()))
            
            wg_label_image = pyift.iftWaterGray(basins, marker, adj1)
            
            dataset = pyift.iftSupervoxelsToSelectiveSearchDataset(self.ift_image, wg_label_image, 8, 0)
            dataset.set_alpha(alpha)
            
            rh = pyift.iftCreateRegionHierarchySelectiveSearch(dataset, adj2)
            label = pyift.iftFlattenRegionHierarchy(rh, int(self.scaler_nsuperpixels.get()))
        else:    
            spatial_radius = float(self.entry_spatial_radius.get())
            volume_threshold = int(self.entry_volume_threshold.get())
            
            seed = pyift.iftLabeledSetFromSeedImage(self.seed_image)
            
            adj_relation = pyift.iftCircular(spatial_radius);
            basins = pyift.iftImageBasins(self.ift_image,adj_relation)
            marker = pyift.iftVolumeClose(basins,volume_threshold)
            label = pyift.iftWaterGray(basins,marker,adj_relation)
            #label = pyift.iftCreateRefinedLabelImage(self.ift_image, seed, spatial_radius, volume_threshold, self.STEPS, self.VOL_RATIO)
        color = pyift.iftColor()
        color.set_values((255,128,128))

        adj_relation = pyift.iftCircular(sqrt(2.))
        adj_relation_2 = pyift.iftCircular(0.0)
        
        ift_image_copy = pyift.iftCopyImage(source)
        pyift.iftDrawBorders(ift_image_copy, label, adj_relation, color, adj_relation_2)
       
        pyift.iftWriteImageP6(ift_image_copy,destination)
        
        if write_label_image:
            pyift.iftWriteImageP2(label,'label.ppm')
        
    def redraw_image(self, event = None):  
        if self.ift_image == None:
            return
        
        if self.show_superpixels.get():
            self.draw_superpixels_on(self.markers_image,"tmp_segmentation.ppm")
            self.update_image("tmp_segmentation.ppm", self.markers_image.xsize, self.markers_image.ysize)     
        else:
            pyift.iftWriteImageP6(self.markers_image,"markers.ppm")
            self.update_image("markers.ppm",self.markers_image.xsize, self.markers_image.ysize)
                    
    def show_classification_mask(self, classification_image):
        classification_mask = pyift.iftCopyImage(self.markers_image)
        
        #Creating classification mask
        #for p in range(0,classification_mask.n,2):
        #    color = self.color_dict[self.color_labels[classification_image[p][0]]]
        #    classification_mask[p] = (classification_mask[p][0],color[1],color[2])
        #F
        color = pyift.iftColor()
        color.set_values(self.color_dict['boundary'])

        adj_relation = pyift.iftCircular(sqrt(2.))
        adj_relation_2 = pyift.iftCircular(1.0)
        
        pyift.iftDrawBorders(classification_mask, classification_image, adj_relation, color, adj_relation_2)

        if self.show_superpixels.get():
            self.draw_superpixels_on(classification_mask,"tmp_class_img.ppm")
        else:
            pyift.iftWriteImageP6(classification_mask,"tmp_class_img.ppm")
            
        self.update_image("tmp_class_img.ppm", classification_mask.xsize, classification_mask.ysize)

    def show_class_classification(self):
        self.save()
        if self.ift_image == None:
            return
        
        classification_image = self.create_classification_cl_image()
        self.show_classification_mask(classification_image)
        
    def show_superpixel_classification(self):
        self.save()
        if self.ift_image == None:
            return
        
        classification_image = self.create_superpixel_cl_image()
        self.show_classification_mask(classification_image)
        
    def show_pixel_classification(self):
        self.save()
        if self.ift_image == None:
            return
        
        classification_image = self.create_pixel_cl_image()
        self.show_classification_mask(classification_image)
        
    def create_classification_cl_image(self):
        #Segmentation
        volume_threshold = int(self.entry_volume_threshold.get())
        vol_decrease = int(volume_threshold/(self.STEPS - 1))  
        
        seed = pyift.iftLabeledSetFromSeedImage(self.seed_image)
        pyift.iftSetTrainingSupervoxelsFromSeeds(self.dataset,self.label_image,seed);
        if self.dataset.ntrainsamples == 0:
            raise Exception("No training samples.")
                
        graph = pyift.iftCreateCplGraph(self.dataset)
        pyift.iftSupTrain(graph)
        pyift.iftClassify(graph,self.dataset)
        
        #Creating classification mask
        classification_image = pyift.iftCreateImage(self.ift_image.xsize,self.ift_image.ysize,self.ift_image.zsize)
       
        for p in range(0,classification_image.n):
            #region label
            region_label = self.label_image[p][0] - 1
            #classification label
            color = self.dataset.get_sample_label(region_label)
            classification_image[p] = (color,0,0)
            
        return classification_image
        
    def create_superpixel_cl_image(self):
        seed = pyift.iftLabeledSetFromSeedImage(self.seed_image)
        pyift.iftSuperpixelClassification(self.region_graph,self.label_image, seed)    
                
        #Creating classification mask
        classification_image = pyift.iftCreateImage(self.ift_image.xsize,self.ift_image.ysize,self.ift_image.zsize)
        for p in range(0,classification_image.n):
            #region label
            region_label = self.label_image[p][0] - 1
            #classification label
            color = self.dataset.get_sample_label(region_label)
            classification_image[p] = (color, 0, 0)
            
        #pyift.iftWriteSeedsOnImage(classification_image,seed)
        
        return classification_image
        
    def create_pixel_cl_image(self):
        a = pyift.iftCircular(3.0)
        
        seed = pyift.iftLabeledSetFromSeedImage(self.seed_image)
        basins = pyift.iftEnhanceEdges(self.ift_image, a, seed, 0.5)
        
        classification_image = pyift.iftWatershed(basins, a, seed)
        #classification_image = pyift.iftWatershedOnPixelDist(self.dataset_pixel,a,seed)
        
        return classification_image

    def update_image(self,image_path, xsize, ysize):
        self.image_canvas.delete(ALL)
        
        self.image_canvas.tk_image = PhotoImage(file=image_path)
        self.image_canvas.image_id = self.image_canvas.create_image(0,0,image = self.image_canvas.tk_image,anchor = NW)
        self.image_canvas.config(scrollregion=(0,0,xsize,ysize))
        self.image_canvas.update()
         
    def create_image_data_structures(self):
        if self.ift_image == None:
            return
        
        #Superpixel dataset and graph
        spatial_radius = float(self.entry_spatial_radius.get())
        volume_threshold = int(self.entry_volume_threshold.get())
        
        adj_relation = pyift.iftCircular(spatial_radius);
        basins = pyift.iftImageBasins(self.ift_image,adj_relation)
        
        marker = pyift.iftVolumeClose(basins,volume_threshold)
        
        self.label_image = pyift.iftWaterGray(basins,marker,adj_relation)
        
        self.dataset = pyift.iftSupervoxelsToDataSet(self.ift_image, self.label_image)
        if self.dataset.nfeats == 3:
            self.dataset.set_alpha([0.2,1.0,1.0])
            
        adj_relation = pyift.iftCircular(1)
        self.region_graph = pyift.iftRegionGraphFromLabelImage(self.label_image,self.dataset,adj_relation)
        
        #Pixel dataset
        self.dataset_pixel = pyift.iftImageToDataSet(self.ift_image)
        
        if self.suggestion_step != "ROOTS":
            self.update_markers_image(redraw_all = True)
            self.redraw_image()
            
        #Suggestion
        self.suggestion_step = "ROOTS"
        self.reducedMST = None
        self.trainGraph = None
        self.reducedZ = None
        self.graph = None

    def _open_image(self,image_path):
        self.master.title("Interactive Segmentation: " + os.path.split(image_path)[1])
        self.ift_image = pyift.iftReadImageP6(image_path)

        self.create_image_data_structures()
        
        current_image_path = os.path.join(self.current_folder,self.current_images[self.current_image_index])
        if os.path.isfile(current_image_path[:-3] + "markers"):
            self._load_markers(current_image_path[:-3] + "markers")
            self.redraw_image()  
        else:
            self.markers_image = pyift.iftCopyImage(self.ift_image)
            self.seed_image = pyift.iftCreateImage(self.ift_image.xsize,self.ift_image.ysize,1)
            pyift.iftSetImage(self.seed_image, -1)
            
            self.redraw_image()  
        
        self.label_total_images.config(text = "/ " + str(len(self.current_images)))
                
    def open_image(self):
        self.save()
        
        image_path = tkFileDialog.askopenfilename(title = "Open Image (.ppm)")
        if image_path == () or image_path == "":
            return
        
        self.current_folder = os.path.split(image_path)[0]
        self.current_images = [os.path.split(image_path)[1]]
        self.current_image_index = 0
        
        self._open_image(image_path)
        self.current_image_var.set(1)

    def export_results(self):
        self.save()
        if self.ift_image == None:
            return
            
        if not os.path.exists(os.path.join(self.current_folder, 'labels')):
            return
        
        filename = tkFileDialog.asksaveasfilename(title = "Export results as (.zip)")
        if filename == () or filename == "":
            return
       
        zipf = zipfile.ZipFile(filename + '.zip', 'w')
        for root, dirs, files in os.walk(os.path.join(self.current_folder, 'labels')):
            for f in files:
                zipf.write(os.path.join(root,f), 'labels/' + f)
        zipf.close()
        #os.system('zip -r {0}.zip {1}'.format(filename,os.path.join(self.current_folder, 'labels')))
        
    def open_image_folder(self):
        self.save()
        
        current_folder = tkFileDialog.askdirectory(title = "Open Image Folder (.ppm images)")
        if current_folder == () or current_folder == '':
            return

        current_images = sorted([f for f in os.listdir(current_folder) if f.endswith(".ppm")])
        if len(current_images) < 1:
            return
        
        self.current_folder = current_folder
        self.current_images = current_images
        self.current_image_index = 0
        
        self._open_image(os.path.join(self.current_folder,self.current_images[0]))
        self.current_image_var.set(1)

    def load_markers(self):
        seed_path = tkFileDialog.askopenfilename()
        if seed_path == () or seed_path == "":
           return
       
        self._load_markers(seed_path)
        self.redraw_image()
       
    def _load_markers(self, seed_path):
       if self.ift_image == None:
           return
       
       seed = pyift.iftReadSeeds2D(seed_path,self.ift_image)
       self.seed_image = pyift.iftSeedImageFromLabeledSet(seed,self.ift_image)
       self.markers_image = pyift.iftCopyImage(self.ift_image)
       
       #creating more labels if needed
       number_labels = pyift.iftMaximumValue(self.seed_image)
       while number_labels >= len(self.color_labels):
           self.add_label() 
        
       self.update_markers_image()
       
    def update_markers_image(self, redraw_all = False):
       if self.ift_image == None:
           return 
        
       for p in range(self.ift_image.n):
            if self.seed_image[p] != (-1,0,0):
                self.markers_image[p] = self.color_dict[self.color_labels[self.seed_image[p][0]]]
            elif redraw_all:
                self.markers_image[p] = self.ift_image[p]
       
    def remove_markers(self):
        if self.ift_image == None:
            return
        
        self.markers_image = pyift.iftCopyImage(self.ift_image)
        self.seed_image = pyift.iftCreateImage(self.ift_image.xsize,self.ift_image.ysize,1)
        pyift.iftSetImage(self.seed_image, -1)
        
        self.redraw_image()
        
        self.suggestion_step = "ROOTS"
        
    def draw_root_suggestion(self,roots):
        roots = roots.to_list()

        for r in roots:
            center_pixel = self.geo_centers[r]
            v = self.ift_image.get_voxel(center_pixel)
            self.draw_circular_marker(1, (v.x,v.y,v.z),float(self.size_spinbox.get())/2 - 2, self.color_dict[self.color_labels[0]], True)
    
    def draw_superpixel_suggestion(self, superpixels):
        superpixels = superpixels.to_list()
        
        for s in superpixels:
            r = self.reducedZ.get_sample_id(s) - 1
            label = self.dataset.get_sample_label(r)
            color = self.color_dict[self.color_labels[label]]
            center_pixel = self.geo_centers[r]
            v = self.ift_image.get_voxel(center_pixel)
            self.draw_circular_marker(-1, (v.x,v.y,v.z), max(1.0,float(self.size_spinbox.get())/2), (255,138,128), False)
            self.draw_circular_marker(label, (v.x,v.y,v.z), max(1.0,float(self.size_spinbox.get())/2 - 2), color, True)
                  
    def handle_superpixel_parameter_change(self, event = None):
        self.create_image_data_structures()
        self.redraw_image()
    
    def show_label_color_dialog(self, update_image = True):
        color = tkColorChooser.askcolor(title="Choose color for Marker \"{}\"".format(self.current_label.get()))[0]
        if color == None:
            return
        
        color = (int(color[0]), int(color[1]), int(color[2]))
        
        ift_color = pyift.iftColor()
        ift_color.set_values(color)
        
        self.color_dict[self.current_label.get()] = pyift.iftRGBtoYCbCr(ift_color).get_values()
        
        self.handle_label_changed()
        
        if update_image:
            self.update_markers_image()

    def show_boundary_color_dialog(self, update_image = True):
        color = tkColorChooser.askcolor(title="Choose color for boundary")[0]
        if color == None:
            return
        
        color = (int(color[0]), int(color[1]), int(color[2]))
        
        ift_color = pyift.iftColor()
        ift_color.set_values(color)
        
        self.color_dict['boundary'] = pyift.iftRGBtoYCbCr(ift_color).get_values()
        
        color = self._convert_ycbcr_to_rgb_str(self.color_dict['boundary'])
        self.button_color.config(bg = color, activebackground=color)
        
        if update_image:
            self.update_markers_image()

    def run(self):
        self.mainloop()
        
    def draw_circular_marker(self, label, pixel, radius, label_color, write_as_seed = True):
        if self.ift_image == None:
            return

        adj_rel = pyift.iftCircular(radius)
        
        (x,y,z) = pixel
        
        for i in range(adj_rel.n):
            d = adj_rel[i]
            adjacent = pyift.iftVoxel()
            (adjacent.x, adjacent.y, adjacent.z) = (x + d[0], y + d[1], z + d[2])
            
            if pyift.iftValidVoxel(self.seed_image, adjacent) == chr(1):
                voxel_index = self.seed_image.get_voxel_index(adjacent)
                if write_as_seed: 
                    self.seed_image[voxel_index] = ( label, 0 , 0 )
                self.markers_image[voxel_index] = label_color
            
        color = self._convert_ycbcr_to_rgb_str(label_color)
        self.image_canvas.create_oval((x,y,x,y),outline=color,width = int(radius*2))
        
    def handle_canvas_click(self,event,label):
        if self.ift_image == None:
            return
        
        pixel = (x,y,z) = ( int(self.image_canvas.canvasx(event.x)), int(self.image_canvas.canvasy(event.y)), 0 )
        radius = float(self.size_spinbox.get())/2
        label_color = self.color_dict[self.color_labels[label]]
        
        self.draw_circular_marker(label, pixel, radius, label_color, True)
        
    def handle_canvas_lclick(self,event):
        self.handle_canvas_click(event, self.color_labels.index(self.current_label.get()))
                    
    def handle_canvas_rclick(self,event):
        self.handle_canvas_click(event,0)
        
    def add_label(self):
        label = "Label {}".format(len(self.color_labels) + 1)
        self.color_labels.append(label)
        
        self.redefine_label_listbox()
        
        self.color_dict[label] = (16,128,128)
        self.current_label.set(label)
        self.show_label_color_dialog(False)
    
    def handle_label_changed(self,*args):
        color = self._convert_ycbcr_to_rgb_str(self.color_dict[self.current_label.get()])
        self.button_color.config(bg = color, activebackground=color)
        
    def redefine_label_listbox(self):
        if self.label_listbox != None:
            self.label_listbox.grid_forget()
            
        self.label_listbox = OptionMenu(self.marker_frame, self.current_label, *self.color_labels )
        self.label_listbox.grid(row = 1, column = 1)
    
    def _convert_ycbcr_to_rgb_str(self,tuple):
        ycbcr_color = pyift.iftColor()
        ycbcr_color.set_values(tuple)
        
        rgb_tuple = pyift.iftYCbCrtoRGB(ycbcr_color).get_values()
        return "#{:02x}{:02x}{:02x}".format(*rgb_tuple)

    def _convert_rgb_to_rgb_str(self, tuple):
        return "#{:02x}{:02x}{:02x}".format(*tuple)

    def save_markers(self):   
        filename = tkFileDialog.asksaveasfilename(title = "Save markers (.txt)")
        if filename == () or filename == "":
            return
        
        self._save_markers(filename)
        
    def _save_markers(self, filename):
        if self.ift_image == None:
            return
        
        seed = pyift.iftLabeledSetFromSeedImage(self.seed_image)
        pyift.iftWriteSeeds2D(filename,seed,self.ift_image)
    
    def save_cl_classification_image(self):
        if self.ift_image == None:
            return
        
        filename = tkFileDialog.asksaveasfilename(title = "Save classification image (.ppm)")
        if filename == () or filename == "":
            return
        
        self.show_class_classification()
        copyfile("tmp_class_img.ppm",filename)
        
    def save_all_superpixels(self):
        if self.ift_image == None:
            return
        
        filename = tkFileDialog.asksaveasfilename(title = "Save image containing all superpixels (.ppm)")
        if filename == () or filename == "":
            return
        
        self.draw_superpixels_on(self.markers_image,"tmp_segmentation.ppm",True)
        copyfile("label.ppm",filename)
        
    def save_super_classification_image(self):
        if self.ift_image == None:
            return
        
        filename = tkFileDialog.asksaveasfilename(title = "Save classification image (.ppm)")
        if filename == () or filename == "":
            return
        
        self.show_superpixel_classification()
        copyfile("tmp_class_img.ppm",filename)
        
    def save_super_mask(self):
        self.save()
        if self.ift_image == None:
            return
        
        filename = tkFileDialog.asksaveasfilename(title = "Save classification image (.pgm)")
        if filename == () or filename == "":
            return
        
        classification_image = self.create_superpixel_cl_image()
        output_image = pyift.iftCreateImage(classification_image.xsize,classification_image.ysize,classification_image.zsize)
        for i in range(0,classification_image.n):
            if(classification_image[i] == (0,0,0)):
                output_image[i] = (0,0,0)
            else:
                output_image[i] = (255,0,0)
        
        pyift.iftWriteImageP5(output_image,filename)
        
    def save_pixel_classification_image(self):
        if self.ift_image == None:
            return
        
        filename = tkFileDialog.asksaveasfilename(title = "Save segmentation image (.ppm)")
        if filename == () or filename == "":
            return
        self.show_pixel_classification()
        copyfile("tmp_class_img.ppm",filename)

    def _save_super_classification(self, filename):
        if self.ift_image == None:
            return
        
        classification_image = self.create_superpixel_cl_image()
        pyift.iftWriteImageP5(classification_image,filename)
        
    def save(self):
        pass

    def handle_button_save(self, *args):
        if self.ift_image == None:
            return
        
        if not os.path.exists(os.path.join(self.current_folder,'labels')):
            os.makedirs(os.path.join(self.current_folder, 'labels'))

        current_image_path = os.path.join(self.current_folder,'labels', self.current_images[self.current_image_index])
        self._save_super_classification(current_image_path[:-3] + 'ppm')
        
    def handle_current_image_change(self,*args):
        self.save()
        
        index = self.current_image_var.get() - 1
        if index >= 0 and index < len(self.current_images):
            self.current_image_index = index
            self._open_image(os.path.join(self.current_folder,self.current_images[index]))
        
    def next_image(self):    
        index = self.current_image_index + 1
        if index >= 0 and index < len(self.current_images):
            self.current_image_var.set(index + 1)
            
        self.handle_current_image_change()
    
    def previous_image(self):
        index = self.current_image_index - 1
        if index >= 0 and index < len(self.current_images):
            self.current_image_var.set(index + 1)
            
        self.handle_current_image_change()

    def next_suggestion(self):
        if self.ift_image == None:
            return

        if self.suggestion_step == "ROOTS":
            #Inverts pixel:superpixel to superpixel:pixel and shift indices by -1
            geo_centers = pyift.iftGeodesicCenters(self.label_image).to_dict()
            self.geo_centers = [None] * len(geo_centers)
            for (k,v) in geo_centers.items():
                self.geo_centers[v-1] = k
            
            #1 = TRAIN
            self.dataset.set_status_all(1)

            self.graph = pyift.iftUnsupLearn3(self.dataset,1.0,self.K_MAX_PROPORTION,10)
            pyift.iftUnsupClassify(self.graph,self.dataset)
            
            #These roots should be labeled by the user
            roots = self.graph.get_roots()
            self.draw_root_suggestion(roots)
            self.suggestion_step = "TRAIN"
            
        elif self.suggestion_step == "TRAIN":
            self.update_markers_image(redraw_all = True)
            self.redraw_image()
            
            seed = pyift.iftLabeledSetFromSeedImage(self.seed_image)
            self.dataset.set_status_all(-1)
            pyift.iftSetTrainingSupervoxelsFromSeeds(self.dataset,self.label_image,seed)
            
            self.trainGraph = pyift.iftCreateCplGraph(self.dataset)
            pyift.iftSupTrain(self.trainGraph)
            pyift.iftClassify(self.trainGraph,self.dataset)
            
            self.reducedZ = pyift.iftGraphBoundaryReduction(self.graph,self.dataset)
            
            self.reducedMST = pyift.iftCreateMST(self.reducedZ)
            pyift.iftNormalizeSampleWeightMST(self.reducedMST)
            
            self.reducedZ.set_status_all(-1)
            self.reducedZ.set_class_all(-1)
            
            #Orders the MST, 0 = IFT_DECREASING
            pyift.iftSortNodesByWeightMST(self.reducedMST,0)
            self.suggestion_step = "MST"
            
        if self.suggestion_step == "MST":
            self.update_markers_image(redraw_all = True)
            self.redraw_image()
            
            seed = pyift.iftLabeledSetFromSeedImage(self.seed_image)
            self.dataset.set_status_all(-1)
            pyift.iftSetTrainingSupervoxelsFromSeeds(self.dataset,self.label_image,seed)
            #also set on reducedz?
            
            self.trainGraph = pyift.iftCreateCplGraph(self.dataset)
            pyift.iftSupTrain(self.trainGraph)
            pyift.iftClassify(self.trainGraph,self.reducedZ)
            
            selectedSamples = pyift.iftSelectSamplesForUserLabelingMST(self.reducedMST,self.SAMPLES_PER_SUGGESTION)
            if selectedSamples:
                self.draw_superpixel_suggestion(selectedSamples)
            
    def quit(self):
        self.save()
        tmp_imgs = ['markers.markers', 'markers.ppm', 'tmp_class_img.ppm', 'tmp_segmentation.ppm']
        for img in tmp_imgs:
            if os.path.exists(img):
                os.remove(img)
        self.master.destroy()
    
if __name__ == '__main__':
    app = InteractiveSegmentationApp(Tk())
    app.run()
    
