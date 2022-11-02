import os

os.environ["OMP_NUM_THREADS"] = "1"

import tifffile
import yaml
import os
from pathlib import Path
from munch import DefaultMunch
import glob
from image_conversion_fun import *

def yaml_to_rt_pipeline(pipeline,output_conf_file,just_demosaicking=False):

    '''

    This function takes as input a pipeline encoded with an object and generates a rawtherapee config file from it.


    Inputs:
    @pipeline : An object containing the config parameters of a pipeline. See below how to transform a yaml config file to such an object.
    @output_conf_file : The output path of the rawtherapee config generated
    @just_demosaicking : A boolean indicating if you just want to demosaick your images without developing them after.

    Example :

    #transform the yaml file into a dictionary
    pipeline=yaml.safe_load(Path('example.yaml').read_text())
    #transform the dictionary into an object with attributes
    pipeline = DefaultMunch.fromDict(pipeline)

    yaml_to_rt_pipeline(pipeline,'./Pipelines/DEVELOPING.pp3',just_demosaicking=False)

    '''
    

    values = dict()
    values["[SharpenMicro]"] = dict()
    values["[Directional Pyramid Denoising]"] = dict()
    values["[Black & White]"] = dict()
    values["[RAW Bayer]"] = dict()
    values["[Resize]"] = dict()
    values["[PostResizeSharpening]"] = dict()
    values["[RAW Bayer]"]["Method"] = pipeline.raw_bayer.method if pipeline.raw_bayer.method else "amaze"

    if not(just_demosaicking):            
        values["[SharpenMicro]"]["Enabled"] = str(pipeline.sharpen_micro.enabled).lower() if pipeline.sharpen_micro.enabled else "false"
        values["[SharpenMicro]"]["Strength"] = str(pipeline.sharpen_micro.strength).lower() if pipeline.sharpen_micro.strength else 1
        values["[SharpenMicro]"]["Uniformity"] = str(pipeline.sharpen_micro.uniformity).lower() if pipeline.sharpen_micro.uniformity else 1

        values["[Directional Pyramid Denoising]"]["Enabled"] = str(pipeline.directional_pyramid_denoising.enabled).lower() if pipeline.directional_pyramid_denoising.enabled else "false"
        values["[Directional Pyramid Denoising]"]["Luma"] = str(pipeline.directional_pyramid_denoising.luma).lower() if pipeline.directional_pyramid_denoising.luma else "0"

        values["[Black & White]"]["Enabled"] = str(pipeline.black_and_white.enabled).lower() if pipeline.black_and_white.enabled else "false"

        values["[Resize]"]["Enabled"] = str(pipeline.resize.enabled).lower() if pipeline.resize.enabled else "false"
        values["[Resize]"]["Width"] = str(pipeline.resize.width) if pipeline.resize.width else "512"
        values["[Resize]"]["Height"] = str(pipeline.resize.height) if pipeline.resize.height else "512"
        values["[Resize]"]["Method"] = pipeline.resize.method if pipeline.resize.method else "Lanczos"
        

        values["[PostResizeSharpening]"]["Enabled"] = str(pipeline.post_resize_sharpening.enabled).lower() if pipeline.post_resize_sharpening.enabled else "false"
        values["[PostResizeSharpening]"]["Radius"] = str(pipeline.post_resize_sharpening.radius).lower() if pipeline.post_resize_sharpening.radius else "0.3"
        values["[PostResizeSharpening]"]["Amount"] = str(pipeline.post_resize_sharpening.amount).lower() if pipeline.post_resize_sharpening.amount else "0"

        
    watch = False
    key = None
    with open(output_conf_file, "w") as fw:
        with open(pipeline.pp3_conf_file, "r") as fr:
            line = fr.readline()
            while line:
                if line[0] == '[':
                    if line[:-1] in values:
                        watch = True
                        key = line[:-1]
                    else:
                        watch = False
                        key = None
                if watch:
                    tmp = line.split('=')[0]
                    if tmp in values[key]:
                        line = tmp + '=' + values[key][tmp] + '\n'
                fw.write(line)
                line = fr.readline()



def develop_raw_pictures(input_folder,output_folder,pipeline_path):
    '''

    The role of this function is to develop raw pictures contained in a specific folder and following a specific pipeline.
    
    Inputs:
    @input_folder : The folder containing your RAW images.
    @output_folder : The folder where you want to see your developed images.
    @pipeline_path : The path of your development pipeline encoded with a yaml file.

    Example : "develop_raw_pictures('Input','Output','example.yaml')

    '''

    #transform the yaml file into a dictionary
    pipeline=yaml.safe_load(Path(pipeline_path).read_text())
    #transform the dictionary into an object with attributes
    pipeline = DefaultMunch.fromDict(pipeline)

    yaml_to_rt_pipeline(pipeline,'./Pipelines/DEMOSAICKING.pp3',just_demosaicking=True)
    yaml_to_rt_pipeline(pipeline,'./Pipelines/DEVELOPING.pp3',just_demosaicking=False)

    #for each RAW picture
    for img in glob.glob(f'{input_folder}/*'):
        
        prefix=img.split('.')[0]

        #First demosaick
        demosaicked_tif_input_file=prefix+"_demosaicked.tif"
        tif_conversion = f"rawtherapee-cli -Y  -o {demosaicked_tif_input_file} -t16 -p ./Pipelines/DEMOSAICKING.pp3  -c {img}" 
        os.system(tif_conversion)

        #Then, "smart" crop is needed
        if  pipeline.crop.enabled:
            cropped_image=edge_crop(tifffile.imread(demosaicked_tif_input_file),1.5,pipeline.crop.h,pipeline.crop.w,64)
        #If crop is "false", we do a squared central crop with the maximum size possible
        else:
            demosaicked_input=tifffile.imread(demosaicked_tif_input_file)
            size=min(demosaicked_input.shape[0],demosaicked_input.shape[1])
            cropped_image=center_crop(demosaicked_input,size,size)

        tifffile.imwrite(demosaicked_tif_input_file,cropped_image)        

        #Developing 
        tif_input_file=prefix+".tif"
        development = "rawtherapee-cli -Y  -o %s -t16 -p %s  -c %s" % (tif_input_file,'./Pipelines/DEVELOPING.pp3',demosaicked_tif_input_file)
    
        os.system(development)

        #Final jpeg compression
        jpeg_compression="convert -define jpeg:optimize-coding=false -quality " + str(pipeline.jpeg.qf) + " " + tif_input_file + " " + prefix.replace(input_folder,output_folder) +".jpg"

        os.system(jpeg_compression)

