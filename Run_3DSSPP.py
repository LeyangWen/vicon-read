import Skeleton
import yaml

config_file = r'F:\wen_storage\test\VEHS_ske\Test\Gunwoo\Test1\Gunwoo movement 02.yaml'

with open(config_file, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
        c3d_file = data['c3d_file']
    except yaml.YAMLError as exc:
        print(config_file, exc)


skeleton = Skeleton.PulginGaitSkeleton(c3d_file)

skeleton.set_weight_height()
# skeleton.set_weight_height(80, 1.844)
skeleton.output_3DSSPP_loc(frame_range=[0,800,5])