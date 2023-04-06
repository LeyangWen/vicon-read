import numpy as np
import xml.etree.ElementTree as ET


class ForceTorqueTransducer:
    def __init__(self):
        pass

class ATIMini45(ForceTorqueTransducer):
    def __init__(self, gain_voltage, zero_frame=(5,20), ft_cal_file=r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\vicon-read\tools\DAQ FT Manual Calculations\FT26836.cal'):
        super().__init__()
        self.ft_cal = self.load_ft_cal(ft_cal_file)
        self.tool_transform = self.load_tool_transform(ft_cal_file)
        self.transformed_ft_cal = self.get_transformed_ft_cal()
        self.zero_frame = zero_frame
        self.gain_voltage = gain_voltage
        self.bias_voltage = np.mean(gain_voltage[zero_frame[0]:zero_frame[1]], axis=0).reshape(1,-1)
        self.force_torque_values = self.convert_to_ft()
        self.__assign_values()

    @staticmethod
    def load_ft_cal(ft_cal_file):
        # Parse the XML file
        tree = ET.parse(ft_cal_file)
        root = tree.getroot()
        # Extract the user axis values
        user_axes = root.findall(".//UserAxis")
        axis_values = []
        for axis in user_axes:
            values = axis.get('values')
            axis_values.append(list(map(float, values.split())))
        # Construct the 6x6 calibration matrix
        ft_cal = np.array(axis_values)
        return ft_cal

    @staticmethod
    def load_tool_transform(ft_cal_file):
        # Parse the XML file
        tree = ET.parse(ft_cal_file)
        root = tree.getroot()
        # Extract the user axis values
        user_axes = root.findall(".//BasicTransform")
        axis_values = []
        for axis in user_axes:
            tool_transform = [axis.get('Dx'), axis.get('Dy'), axis.get('Dz'), axis.get('Rx'), axis.get('Ry'), axis.get('Rz')]
            tool_transform = list(map(float, tool_transform))
            tool_transform = np.array(tool_transform)
            return tool_transform
        raise ValueError(f'No tool transform found in calibration file {ft_cal_file}')

    def convert_to_ft(self):
        adjusted_voltages = self.gain_voltage - self.bias_voltage # Subtract the reference values from the voltage readings
        adjusted_voltages = adjusted_voltages.reshape(-1, 7) # Reshape the voltage readings to a 1x7 array
        temp_voltage = adjusted_voltages[:,-1]
        adjusted_voltages = adjusted_voltages[:,:-1] # Remove the last value from the voltage readings
        force_torque_values = self.transformed_ft_cal.dot(adjusted_voltages.T).T # Apply the calibration values to convert voltage readings to force and torque values
        return force_torque_values

    def get_transformed_ft_cal(self):
        # Apply the tool transform to the force and torque values
        Dx, Dy, Dz, Rx, Ry, Rz = self.tool_transform
        rot_matrix = np.zeros((6, 6))
        rot_matrix[0, 0] = np.cos(Ry) * np.cos(Rz)
        rot_matrix[0, 1] = np.sin(Rx) * np.sin(Ry) * np.cos(Rz) + np.cos(Rx) * np.sin(Rz)
        rot_matrix[0, 2] = np.sin(Rx) * np.sin(Rz) - np.cos(Rx) * np.sin(Ry) * np.cos(Rz)
        rot_matrix[1, 0] = (-1) * np.cos(Ry) * np.sin(Rz)
        rot_matrix[1, 1] = (-1) * np.sin(Rx) * np.sin(Ry) * np.sin(Rz) + np.cos(Rx) * np.cos(Rz)
        rot_matrix[1, 2] = np.sin(Rx) * np.cos(Rz) + np.cos(Rx) * np.sin(Ry) * np.sin(Rz)
        rot_matrix[2, 0] = np.sin(Ry)
        rot_matrix[2, 1] = (-1) * np.sin(Rx) * np.cos(Ry)
        rot_matrix[2, 2] = np.cos(Rx) * np.cos(Ry)
        rot_matrix[3, 3] = np.cos(Ry) * np.cos(Rz)
        rot_matrix[3, 4] = np.sin(Rx) * np.sin(Ry) * np.cos(Rz) + np.cos(Rx) * np.sin(Rz)
        rot_matrix[3, 5] = np.sin(Rx) * np.sin(Rz) - np.cos(Rx) * np.sin(Ry) * np.cos(Rz)
        rot_matrix[4, 3] = (-1) * np.cos(Ry) * np.sin(Rz)
        rot_matrix[4, 4] = (-1) * np.sin(Rx) * np.sin(Ry) * np.sin(Rz) + np.cos(Rx) * np.cos(Rz)
        rot_matrix[4, 5] = np.sin(Rx) * np.cos(Rz) + np.cos(Rx) * np.sin(Ry) * np.sin(Rz)
        rot_matrix[5, 3] = np.sin(Ry)
        rot_matrix[5, 4] = (-1) * np.sin(Rx) * np.cos(Ry)
        rot_matrix[5, 5] = np.cos(Rx) * np.cos(Ry)

        translation_matrix = np.eye(6)
        translation_matrix[4, 2] = Dx
        translation_matrix[5, 0] = Dy
        translation_matrix[3, 1] = Dz
        translation_matrix[5, 1] = (-1) * Dx
        translation_matrix[3, 2] = (-1) * Dy
        translation_matrix[4, 0] = (-1) * Dz

        transformed_ft_cal = rot_matrix.dot(translation_matrix).dot(self.ft_cal)
        # print(rot_matrix)
        # print(translation_matrix)
        # print(transformed_ft_cal)
        return transformed_ft_cal

    def __assign_values(self):
        self.force = self.force_torque_values[:,:3]
        self.torque = self.force_torque_values[:,3:]
        self.Fx = self.force[:,0]
        self.Fy = self.force[:,1]
        self.Fz = self.force[:,2]
        self.Tx = self.torque[:,0]
        self.Ty = self.torque[:,1]
        self.Tz = self.torque[:,2]

    def temp_compensation(self):
        raise NotImplementedError('Temperature compensation not implemented because temperature slops are not in the calibration file')
        # thermistor = 0.000000
        # bias_slopes = [0, 0, 0, 0, 0, 0, 0]
        # gain_slopes = [0, 0, 0, 0, 0, 0, 0]

    @staticmethod
    def conversion_test():
        transducer = ForceTorqueTransducer()
        transducer.ft_cal = np.array([[0.211752627, -0.054120675, -1.387840088, 13.78393409, 0.648967015, -13.14079019],
                                      [0.001747318, -16.01448182, -0.983632307, 7.960238441, -0.126542006, 7.537487536],
                                      [26.52834231, -0.14476068, 25.5133154, -0.576553959, 25.66105485, 0.218172629],
                                      [0.006053029, -0.134314159, 0.241133362, 0.062793403, -0.250674306, 0.059019675],
                                      [-0.305874895, 0.001082498, 0.158667919, -0.119854119, 0.134899934, 0.11324404],
                                      [-0.000345741, -0.128724194, 0.007221646, -0.125141496, 0.009693415,-0.119365973]])
        transducer.bias_voltage = np.array([[0.8235, 0.5667, -0.3255, 0.1217, -0.247, 0.7753, 2.2927],[0.8235, 0.5667, -0.3255, 0.1217, -0.247, 0.7753, 2.2927]])
        transducer.gain_voltage = np.array([[0.1099, 1.9644, 3.694, -1.0388, -3.8624, -0.7913, 2.5136],[0.1099, 1.9644, 3.694, -1.0388, -3.8624, -0.7913, 2.5136]])
        transducer.tool_transform = np.zeros(6)
        transducer.transformed_ft_cal = transducer.get_transformed_ft_cal()
        transducer.force_torque_values = transducer.convert_to_ft()
        # Fx	Fy	Fz	Tx	Ty	Tz
        # -3.561343331	-46.92698291	-9.029861915	1.518141126	0.331516401	0.146536188
        print(f"correct values: -3.561343331	-46.92698291	-9.029861915	1.518141126	0.331516401	0.146536188")
        print(f"calculated values: {transducer.force_torque_values}")
