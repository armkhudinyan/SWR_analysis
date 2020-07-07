import datetime as dt
print(f'[{dt.datetime.now()}] Initializing Active Learning Iteration...')

import os
import sys
from subprocess import Popen, check_call

import optparse
optparser = optparse.OptionParser()
optparser.add_option(
    "--n_run",
    help = "Mandatory parameter. Iteration number."
)

optparser.add_option(
    "--n_patches", default = '10',
    help = "Number of patches per class to output in the final uncertainty patches passed to photointerpreters."
)

optparser.add_option(
    "--composite_path", default = r'\\dgt-759\S2_2018\Theia_S2process\T29SND\composites',
    help = "Tile composite directory."
)

options = optparser.parse_args()[0]
if options.n_run is None:   # if filename is not given
    optparser.error('Mandatory argument n_run not given.')

n_run = int(options.n_run)
n_patches = int(options.n_patches)
composite_path = options.composite_path

# =========================================================================
# Sanity check
# =========================================================================
# check wether n_run is overwritting other runs and/or if it can be run
if os.path.isfile(os.path.join(os.path.dirname(__file__),'truth_rasters',f'truth_patches_{n_run-1}.tif')):
    while True:
        repeat_run = input(f"""
        It appears that iteration {n_run} was already run in the past, be it successfully or not.
        Proceeding means you will be overwritting existing files/results.
        Would you still like to proceed [y/n]? """)
        if repeat_run.lower() in ['no', 'n']:
            print('Goodbye.')
            sys.exit()
        elif repeat_run.lower() in ['yes', 'ye', 'y']:
            break
        else:
            print('Invalid answer.')

scripts_path = os.path.join(os.path.dirname(__file__),'scripts')
# =========================================================================
# run pipeline
# =========================================================================

# 1) convert to raster
print(f'\n[{dt.datetime.now()}] [1/5] Rasterizing photointerpreted patches from previous interation...')
# run f'rasterize_truth_data.py --n_run={n_run-1}'
script = os.path.join(scripts_path, 'rasterize_truth_data.py')
command = Popen(f'python {script} --n_run={n_run-1}', shell=True).wait()
check_call(command)




# 2) create training dataset
print(f'[{dt.datetime.now()}] [2/5] Creating training dataset...')
# run f'train_data_resampling.py --n_run={n_run} --composite_path=composite_path'
script = os.path.join(scripts_path, 'train_data_resampling.py')
command = Popen(f'python {script} --n_run={n_run} --composite_path={composite_path}', shell=True).wait()
check_call(command)



# 3) main active learning procedure
print(f'[{dt.datetime.now()}] [3/5] Running main active learning procedure -> AL_procedure.py --n_run={n_run} --composite_path={composite_path}')
# run f'AL_procedure.py --n_run={n_run} --composite_path={composite_path}'
script = os.path.join(scripts_path, 'AL_procedure.py')
command = Popen(f'python {script} --n_run={n_run} --composite_path={composite_path}', shell=True).wait()
check_call(command)



# 4) ArcGIS Toolbox
print(f"""
[{dt.datetime.now()}] [4/5] Main AL procedure finished. Results were stored in AL_T29SND/results/

Uncertainty map post-processing step reached. Please open ArcMAP, import the toolbox provided matching your ArcMAP (ArcGIS v>=10.4)  and follow chapter 3 of AL_implementation_documentation.pdf

NOTES:
    1) Save the output shapefile in the following path: AL_T29SND/uncertainty_intersect/uncert_intersect_{n_run}.shp
    2) Do not cancel the process at this point, but only proceed after having completed the current step.

""")
while True:
    uncertainty_map_path = input("""Please introduce the path for the saved shapefile (.shp): """)
    if not uncertainty_map_path.endswith('.shp'):
        print('Invalid path, must point to a .shp file.')
    elif not os.path.isfile(uncertainty_map_path):
        print('Path doesn\'t exist.')
    else:
        break




# 5) large uncertainty patches selection
print(f"""[{dt.datetime.now()}] [5/5] Running large uncertainty patches selection step -> large_patch_selection.py  --n_run={n_run} --n_patches={n_patches} --uncertainty_path={uncertainty_map_path}""")
# run f'large_patch_selection.py --n_run={n_run} --n_patches={n_patches} --uncertainty_path={uncertainty_map_path}'
script = os.path.join(scripts_path, 'large_patch_selection.py')
command = Popen(f'python {script} --n_run={n_run} --n_patches={n_patches} --uncertainty_path={uncertainty_map_path}', shell=True).wait()
check_call(command)

print(f"""[{dt.datetime.now()}] Process complete. You can find the uncertainty patches to provide to photointerpreters in AL_T29SND/uncertainty_patches""")
