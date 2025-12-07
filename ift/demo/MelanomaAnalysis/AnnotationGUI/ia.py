#!/usr/bin/python2.7
import pyift
from Tkinter import *
import tkFileDialog
import tkColorChooser
import tkSimpleDialog
import os
import zipfile
import json
import re
from skimage import segmentation
from skimage import io

class InteractiveAnnotationApp(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        master.title("Interactive Annotation")
        
        self.color_labels = ["No label"]
        self.color_dict = { self.color_labels[0] : (125, 128, 128) }
         
        self.ift_image = None
        self.ift_image_segmented = None
        self.seed_image = None
        self.conf_image = None
        self.markers_image = None
         
        self.current_folder = ""
        self.current_images = []
        self.current_image_index = 0
         
        self.OVERSEG_COLOR = (255,128,128)
        
        self.COMPACTNESS = 20.0
        
        self.create_widgets()
         
    def create_widgets(self):
        self.pack(expand = True, fill = IFT_BOTH)
         
        #Menu
        self.menubar = Menu(self.master)
         
        filemenu = Menu(self.menubar, tearoff=0)
        filemenu.add_command(label = "Load folder...", command = self.handle_load_folder)
        filemenu.add_command(label = 'Export annotations...', command = self.handle_export_annotations)
        filemenu.add_command(label = 'Quit', command = self.handle_quit)
         
        self.menubar.add_cascade(label = "File", menu = filemenu)
         
        self.master.config(menu=self.menubar)
         
        #Left Frame
        self.left_frame = Frame(self, relief = RAISED, borderwidth = 1)
        self.left_frame.pack(side = LEFT, anchor = NE, expand = False)
         
        #Markers Frame
        self.marker_frame = LabelFrame(self.left_frame, text = "Marker")
        self.marker_frame.pack(side = TOP,fill = X);
         
        size_label = Label(self.marker_frame, text = "Size:")
        size_label.grid(row = 0, column = 0)
        self.size_spinbox = Spinbox(self.marker_frame, from_ = 3, to = 50, width = 2)
        self.size_spinbox.delete(0,END)
        self.size_spinbox.insert(0,"5")
        self.size_spinbox.grid(row = 0, column = 1, columnspan = 2)
         
        color = '#545454'
        self.button_color = Button(self.marker_frame, bg = color, activebackground=color, relief = RAISED, borderwidth = 1, command = self.handle_change_marker_color)
        self.button_color.grid(row=2,column = 0)
        
        self.button_rename_label = Button(self.marker_frame, text = "...", command = self.handle_rename_label)
        self.button_rename_label.grid(row = 2, column = 1)
        self.button_add_label = Button(self.marker_frame, text = "+", command = self.handle_add_label)
        self.button_add_label.grid(row = 2, column = 2)
        
        self.current_label = StringVar(self.marker_frame)
        self.current_label.set(self.color_labels[0])
        self.current_label.trace("w", self.handle_label_changed)
        self.label_listbox = None
        self.redefine_label_listbox()
        self.handle_label_changed()
        
        confidence_label = Label(self.marker_frame, text = "Confidence:")
        confidence_label.grid(row=3,column = 0)
        self.entry_confidence = Entry(self.marker_frame, width = 3)
        self.entry_confidence.insert(0, "100")
        self.entry_confidence.grid(row = 3, column = 1, columnspan = 2)
        
        self.annotation_frame = LabelFrame(self.left_frame, text = "Annotation")
        self.annotation_frame.pack(side = TOP, fill = X)
        
        self.button_update = Button(self.annotation_frame, text = "Update coloring", command = self.handle_update_image)
        self.button_update.pack(side = TOP)
        
        self.button_reload = Button(self.annotation_frame, text = "Revert", command = self.handle_revert)
        self.button_reload.pack(side = TOP)
     
        self.button_remove_all = Button(self.annotation_frame, text = "Remove all", command = self.handle_remove_all)
        self.button_remove_all.pack(side = TOP)
        
        #Oversegmentation Frame
        self.overseg_frame = LabelFrame(self.left_frame, text = "Oversegmentation")
        self.overseg_frame.pack(side = TOP,fill = X)
        
        label_superpixels = Label(self.overseg_frame, text = "Superpixels:")
        label_superpixels.grid(row = 0, column = 0)
        self.entry_superpixels = Entry(self.overseg_frame, width = 5)
        self.entry_superpixels.insert(0, "300")
        self.entry_superpixels.bind("<Return>",self.handle_superpixel_parameter_change)
        self.entry_superpixels.grid(row = 0, column = 1)
        
        self.button_ok_parameters = Button(self.overseg_frame, text="OK",command=self.handle_superpixel_parameter_change)
        self.button_ok_parameters.grid(row=0,column = 2)
        
        self.show_superpixels = IntVar()
        self.show_superpixels.set(1)
        self.toggle_superpixels = Checkbutton(self.overseg_frame, text="View superpixels", variable=self.show_superpixels, command=self.handle_update_image)
        self.toggle_superpixels.grid(row = 1, column = 0, columnspan = 3)
        
        self.show_segmentation = IntVar()
        self.show_segmentation.set(1)
        self.toggle_segmentation = Checkbutton(self.overseg_frame, text="View segmentation", variable=self.show_segmentation, command=self.handle_update_image)
        self.toggle_segmentation.grid(row = 2, column = 0, columnspan = 3)
        
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
        self.button_previous = Button(self.navigation_frame,text = "< Previous",command = self.handle_previous_image)
        self.button_previous.pack(side = LEFT,padx = 3)
        
        self.current_image_var = IntVar(self.navigation_frame)
        self.current_image_var.set(0)
        self.entry_current_image = Entry(self.navigation_frame,width = 5,textvariable = self.current_image_var)
        self.entry_current_image.pack(side = LEFT,anchor = CENTER)
        self.entry_current_image.bind("<Return>",self.handle_current_image_change)
        
        self.label_total_images = Label(self.navigation_frame, text = "/ 0")
        self.label_total_images.pack(side = LEFT)
        self.button_next = Button(self.navigation_frame, text = "Next >", command = self.handle_next_image)
        self.button_next.pack(side = LEFT,padx = 3)
        
        #Mouse Events
        self.image_canvas.bind("<B1-Motion>", self.handle_canvas_lclick)
        self.image_canvas.bind("<Button-1>", self.handle_canvas_lclick)

        #Other Events:
        self.master.protocol("WM_DELETE_WINDOW",self.handle_quit)
     
    def handle_current_image_change(self, event = None):
        self.save()
        
        index = self.current_image_var.get() - 1
        if index >= 0 and index < len(self.current_images):
            self.current_image_index = index
            self._open_image()
        
    def handle_previous_image(self):
        index = self.current_image_index - 1
        if index >= 0 and index < len(self.current_images):
            self.current_image_var.set(index + 1)
            
        self.handle_current_image_change()
    
    def handle_next_image(self):
        index = self.current_image_index + 1
        if index >= 0 and index < len(self.current_images):
            self.current_image_var.set(index + 1)
            
        self.handle_current_image_change()
    
    def draw_segmentation_on(self, source, destination):
        color = pyift.iftColor()
        color.set_values(self.OVERSEG_COLOR)

        adj_relation = pyift.iftCircular(1.5)
        adj_relation_2 = pyift.iftCircular(0.0)
        
        ift_image_copy = pyift.iftCopyImage(source)
        pyift.iftDrawBorders(ift_image_copy, self.ift_image_segmented, adj_relation, color, adj_relation_2)
        pyift.iftWriteImageP6(ift_image_copy,destination)
    
    def draw_superpixels_on(self, source, destination):
        color = pyift.iftColor()
        color.set_values(self.OVERSEG_COLOR)

        adj_relation = pyift.iftCircular(1.5)
        adj_relation_2 = pyift.iftCircular(0.0)
        
        ift_image_copy = pyift.iftCopyImage(source)
        #Draw over image the color representing annotated pixels, if any
        seed = pyift.iftLabeledSetFromSeedImage(self.seed_image)
        if seed:
            seed_dict = seed.to_dict()
            region_color = {}
            for p,l in seed_dict.iteritems():
                region_color[self.label_image[p]] = l
            
            for p in range(0,ift_image_copy.n,2):
                if self.label_image[p] in region_color:
                    c = self.color_dict[self.color_labels[region_color[self.label_image[p]]]]
                    ift_image_copy[p] = (ift_image_copy[p][0],c[1],c[2])
        
        pyift.iftDrawBorders(ift_image_copy, self.label_image, adj_relation, color, adj_relation_2)
        pyift.iftWriteImageP6(ift_image_copy,destination)
     
    def handle_update_image(self): 
        if self.ift_image == None:
            return
        
        if self.show_superpixels.get():
            self.draw_superpixels_on(self.markers_image,"tmp_segmentation.ppm")
            self.update_image("tmp_segmentation.ppm", self.markers_image.xsize, self.markers_image.ysize)
        elif self.show_segmentation.get():
            self.draw_segmentation_on(self.markers_image, "tmp_segmentation.ppm")
            self.update_image("tmp_segmentation.ppm", self.markers_image.xsize, self.markers_image.ysize)
        else:
            pyift.iftWriteImageP6(self.markers_image,"markers.ppm")
            self.update_image("markers.ppm",self.markers_image.xsize, self.markers_image.ysize)
            
    def update_image(self, image_path, xsize, ysize):
        self.image_canvas.delete(ALL)
        
        self.image_canvas.tk_image = PhotoImage(file=image_path)
        self.image_canvas.image_id = self.image_canvas.create_image(0,0,image = self.image_canvas.tk_image,anchor = NW)
        self.image_canvas.config(scrollregion=(0,0,xsize,ysize))
        self.image_canvas.update()
    
    def _create_image_data_structures(self):
        if self.ift_image == None:
            return
        
        adj_relabel = pyift.iftCircular(1.5)
        
        image_path = os.path.join(self.current_folder,self.current_images[self.current_image_index])
        im = io.imread(image_path)
        region_labels = segmentation.slic(im, n_segments = int(self.entry_superpixels.get()), compactness = self.COMPACTNESS,
                                          enforce_connectivity = True)
                                                  
        self.label_image = pyift.iftCreateImage(region_labels.shape[1], region_labels.shape[0], 1)
        pyift.setIftImage(self.label_image, list(region_labels.reshape(-1) + 1))
        #Masking
        self.label_image = pyift.iftMask(self.label_image, self.ift_image_segmented)
        self.label_image = pyift.iftRelabelRegions(self.label_image, adj_relabel)
        self.label_image.add_const(1)

        #Watershed superpixels
        # adj_relabel = pyift.iftCircular(1.5)
        # 
        # #Superpixel dataset and graph
        # spatial_radius = float(self.SPATIAL_RADIUS)
        # volume_threshold = int(self.entry_superpixels.get())
        # 
        # adj_relation = pyift.iftCircular(spatial_radius);
        # basins = pyift.iftImageBasins(self.ift_image,adj_relation)
        # marker = pyift.iftVolumeClose(basins,volume_threshold)
        # 
        # self.label_image = pyift.iftWaterGray(basins,marker,adj_relation)
        # #Masking
        # self.label_image = pyift.iftMask(self.label_image, self.ift_image_segmented)
        # self.label_image = pyift.iftRelabelRegions(self.label_image, adj_relabel)
        # self.label_image.add_const(1)
    
    def _open_image(self):
        image_path = os.path.join(self.current_folder,self.current_images[self.current_image_index])
        self.master.title("Interactive Annotation: " + os.path.split(image_path)[1])
        self.ift_image = pyift.iftReadImageP6(image_path)
        
        base_path = os.path.split(os.path.split(image_path)[0])[0] 
        self.ift_image_segmented = pyift.iftReadImageP5(os.path.join(base_path, 'labels', os.path.split(image_path)[1][:-3] + 'pgm'))

        if os.path.exists(os.path.join(os.path.split(image_path)[0],'overseg', os.path.split(image_path)[1][:-3] + 'pgm')):
            self.label_image = pyift.iftReadImageP5(os.path.join(os.path.split(image_path)[0],'overseg', os.path.split(image_path)[1][:-3] + 'pgm'))
            
            seed = pyift.iftReadSeeds2D(os.path.join(os.path.split(image_path)[0],'markers', os.path.split(image_path)[1][:-3] + 'txt'),self.ift_image)
            self.seed_image = pyift.iftSeedImageFromLabeledSet(seed,self.ift_image)
            
            conf = pyift.iftReadSeeds2D(os.path.join(os.path.split(image_path)[0],'confidence', os.path.split(image_path)[1][:-3] + 'txt'),self.ift_image)
            self.conf_image = pyift.iftSeedImageFromLabeledSet(conf,self.ift_image)
            
            self.markers_image = pyift.iftCopyImage(self.ift_image)
            self.update_markers_image()
            
            params_file = open(os.path.join(os.path.split(image_path)[0],'params', os.path.split(image_path)[1][:-3] + 'txt'),'r')
            params = params_file.read()
            
            p = re.compile('superpixels [0-9]+')
            vt = p.findall(params)[0].split()[1]
            self.entry_superpixels.delete(0, END)
            self.entry_superpixels.insert(0, vt)
        else:
            self._create_image_data_structures()
            
            self.markers_image = pyift.iftCopyImage(self.ift_image)
            self.seed_image = pyift.iftCreateImage(self.ift_image.xsize,self.ift_image.ysize,1)
            pyift.iftSetImage(self.seed_image, -1)
            self.conf_image = pyift.iftCreateImage(self.ift_image.xsize,self.ift_image.ysize,1)
            pyift.iftSetImage(self.conf_image, -1)
        
        self.handle_update_image()  
        self.label_total_images.config(text = "/ " + str(len(self.current_images)))
    
    def handle_load_folder(self):    
        self.save()
        
        current_folder = tkFileDialog.askdirectory(title = "Load folder (.ppm images)")
        if current_folder == () or current_folder == '':
            return

        current_images = sorted([f for f in os.listdir(current_folder) if f.endswith(".ppm")])
        if len(current_images) < 1:
            return
        
        self.current_folder = current_folder
        self.current_images = current_images
        self.current_image_index = 0
        
        if os.path.exists(os.path.join(self.current_folder, 'params', 'labels.txt')):
            self.color_labels = json.load(open(os.path.join(self.current_folder, 'params', 'labels.txt'), 'r'))
            self.color_dict = json.load(open(os.path.join(self.current_folder, 'params', 'colors_labels.txt'), 'r'))
            for k in self.color_dict.keys():
                self.color_dict[k] = tuple(self.color_dict[k])
            
            last_image = json.load(open(os.path.join(self.current_folder, 'params', 'last.txt'), 'r'))
             
            try:
                self.current_image_index = self.current_images.index(last_image)
            except:
                self.current_image_index = 0
            
            self.current_label.set(self.color_labels[0])    
            self.redefine_label_listbox()
            self.handle_label_changed()
                
        self._open_image()
        self.current_image_var.set(self.current_image_index + 1)
        
    def handle_export_annotations(self):
        self.save()
       
        if self.ift_image == None:
            return
             
        if not os.path.exists(os.path.join(self.current_folder, 'overseg')):
            return
         
        filename = tkFileDialog.asksaveasfilename(title = "Export results as (.zip)")
        if filename == () or filename == "":
            return
        
        if not filename.endswith('.zip'):
            filename += '.zip'
         
        zipf = zipfile.ZipFile(filename, 'w')
        
        for root, dirs, files in os.walk(os.path.join(self.current_folder, 'overseg')):
            for f in files:
                zipf.write(os.path.join(root,f), 'overseg/' + f)
                
        for root, dirs, files in os.walk(os.path.join(self.current_folder, 'markers')):
            for f in files:
                zipf.write(os.path.join(root,f), 'markers/' + f)
                
        for root, dirs, files in os.walk(os.path.join(self.current_folder, 'confidence')):
            for f in files:
                zipf.write(os.path.join(root,f), 'confidence/' + f)
                
        for root, dirs, files in os.walk(os.path.join(self.current_folder, 'params')):
            for f in files:
                zipf.write(os.path.join(root,f), 'params/' + f)

        zipf.close()
     
    def handle_change_marker_color(self):
        color = None
        while color == None:
            color = tkColorChooser.askcolor(title="Choose color for Marker \"{}\"".format(self.current_label.get()))[0]
        
        color = (int(color[0]), int(color[1]), int(color[2]))
        
        ift_color = pyift.iftColor()
        ift_color.set_values(color)
        
        self.color_dict[self.current_label.get()] = pyift.iftRGBtoYCbCr(ift_color).get_values()
        
        self.handle_label_changed()
        
        self.update_markers_image()
        
    def update_markers_image(self):
       if self.ift_image == None:
           return 
        
       for p in range(self.ift_image.n):
            if self.seed_image[p] != (-1,0,0):
                self.markers_image[p] = self.color_dict[self.color_labels[self.seed_image[p][0]]]
    
    def handle_superpixel_parameter_change(self, event = None):
        self._create_image_data_structures()
        self.handle_update_image()
    
    def handle_label_changed(self,*args):
        color = self._convert_ycbcr_to_rgb_str(self.color_dict[self.current_label.get()])
        self.button_color.config(bg = color, activebackground=color)
        
    def redefine_label_listbox(self):
        if self.label_listbox != None:
            self.label_listbox.grid_forget()
            
        self.label_listbox = OptionMenu(self.marker_frame, self.current_label, *self.color_labels)
        self.label_listbox.grid(row = 1, column = 0, columnspan = 3)
        self.label_listbox.config(width = 15)
        
    def handle_remove_all(self):
        if self.ift_image == None:
            return
        
        self.markers_image = pyift.iftCopyImage(self.ift_image)
        self.seed_image = pyift.iftCreateImage(self.ift_image.xsize,self.ift_image.ysize,1)
        pyift.iftSetImage(self.seed_image, -1)
        self.conf_image = pyift.iftCreateImage(self.ift_image.xsize,self.ift_image.ysize,1)
        pyift.iftSetImage(self.conf_image, -1)
        
        self.handle_update_image()

    def handle_canvas_lclick(self, event):
        label = self.color_labels.index(self.current_label.get())
        if self.ift_image == None:
            return
        
        pixel = (x,y,z) = ( int(self.image_canvas.canvasx(event.x)), int(self.image_canvas.canvasy(event.y)), 0 )
        radius = float(self.size_spinbox.get())/2
        label_color = self.color_dict[self.color_labels[label]]
        
        self.draw_circular_marker(label, pixel, radius, label_color)
        
    def draw_circular_marker(self, label, pixel, radius, label_color):
        if self.ift_image == None:
            return

        adj_rel = pyift.iftCircular(radius)
        
        (x,y,z) = pixel
        confidence = int(self.entry_confidence.get())
        
        for i in range(adj_rel.n):
            d = adj_rel[i]
            adjacent = pyift.iftVoxel()
            (adjacent.x, adjacent.y, adjacent.z) = (x + d[0], y + d[1], z + d[2])
            
            if pyift.iftValidVoxel(self.seed_image, adjacent) == chr(1):
                voxel_index = self.seed_image.get_voxel_index(adjacent)
                self.seed_image[voxel_index] = ( label, 0 , 0 )
                self.conf_image[voxel_index] = (confidence, 0 , 0)
                self.markers_image[voxel_index] = label_color
            
        color = self._convert_ycbcr_to_rgb_str(label_color)
        self.image_canvas.create_oval((x,y,x,y),outline=color,width = int(radius*2))
    
    def handle_rename_label(self):
        old_name = self.current_label.get()
        
        new_name = None
        while new_name == None or new_name in self.color_labels:
            new_name = tkSimpleDialog.askstring("Edit label", 'Name:')
        
        self.color_dict[new_name] = self.color_dict[old_name]
        
        ind = self.color_labels.index(old_name)
        self.color_labels[ind] = new_name
        
        del self.color_dict[old_name]
        self.redefine_label_listbox()
        
        self.current_label.set(new_name)
    
    def handle_add_label(self):
        label = None
        while label == None or label in self.color_labels:
            label = tkSimpleDialog.askstring("New label", 'Name:')
            
        self.color_labels.append(label)
        
        self.redefine_label_listbox()
        
        self.color_dict[label] = (16,128,128)
        self.current_label.set(label)
        self.handle_change_marker_color()
    
    def save(self):
        if self.ift_image == None:
            return
        
        if not os.path.exists(os.path.join(self.current_folder,'overseg')):
            os.makedirs(os.path.join(self.current_folder, 'overseg'))
        if not os.path.exists(os.path.join(self.current_folder,'markers')):
            os.makedirs(os.path.join(self.current_folder, 'markers'))
        if not os.path.exists(os.path.join(self.current_folder,'confidence')):
            os.makedirs(os.path.join(self.current_folder, 'confidence'))
        if not os.path.exists(os.path.join(self.current_folder,'params')):
            os.makedirs(os.path.join(self.current_folder, 'params'))
        
        overseg_path = os.path.join(self.current_folder,'overseg', self.current_images[self.current_image_index])
        overseg_path = overseg_path[:-3] + 'pgm'
        
        seeds_path = os.path.join(self.current_folder,'markers', self.current_images[self.current_image_index])
        seeds_path = seeds_path[:-3] + 'txt'
        
        confidence_path = os.path.join(self.current_folder,'confidence', self.current_images[self.current_image_index])
        confidence_path = confidence_path[:-3] + 'txt'
        
        params_path = os.path.join(self.current_folder,'params', self.current_images[self.current_image_index])
        params_path = params_path[:-3] + 'txt'
                
        pyift.iftWriteImageP5(self.label_image,overseg_path)

        seed = pyift.iftLabeledSetFromSeedImage(self.seed_image)
        pyift.iftWriteSeeds2D(seeds_path, seed, self.seed_image)

        seed = pyift.iftLabeledSetFromSeedImage(self.conf_image)
        pyift.iftWriteSeeds2D(confidence_path, seed, self.conf_image)
        
        f = open(params_path, 'w')
        f.write('superpixels {0}\n'.format(int(self.entry_superpixels.get())))
        f.close()
        
        f = open(os.path.join(self.current_folder, 'params','labels.txt'), 'w')
        json.dump(self.color_labels,f)
        f.close()
        
        f = open(os.path.join(self.current_folder, 'params','colors_labels.txt'), 'w')
        json.dump(self.color_dict,f)
        f.close()
        
        f = open(os.path.join(self.current_folder, 'params', 'last.txt'), 'w')
        json.dump(self.current_images[self.current_image_index], f)
        f.close()
    
    def handle_revert(self):
        if self.ift_image == None:
            return
        
        self._open_image()
    
    def handle_quit(self):
        self.save()
        
        tmp_imgs = ['markers.ppm', 'tmp_segmentation.ppm']
        for img in tmp_imgs:
            if os.path.exists(img):
                os.remove(img)
                
        self.master.destroy()
        
    def run(self):
        self.mainloop()
        
    def _convert_ycbcr_to_rgb_str(self,tuple):
        ycbcr_color = pyift.iftColor()
        ycbcr_color.set_values(tuple)
        
        rgb_tuple = pyift.iftYCbCrtoRGB(ycbcr_color).get_values()
        return "#{:02x}{:02x}{:02x}".format(*rgb_tuple)

if __name__ == '__main__':
    app = InteractiveAnnotationApp(Tk())
    app.run()