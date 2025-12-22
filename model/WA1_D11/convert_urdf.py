
import mujoco
import os

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the input and output file paths relative to the script directory
urdf_filename = os.path.join(script_dir, 'urdf/WA1_D11_old.urdf')
xml_filename = os.path.join(script_dir, 'urdf/WA1_D11.xml')

# Load the URDF file
try:
    model = mujoco.MjModel.from_xml_path(urdf_filename)
    print(f"Successfully loaded URDF from: {urdf_filename}")

    # Save the model to an MJCF XML file
    mujoco.mj_saveLastXML(xml_filename, model)
    print(f"Successfully converted and saved MJCF to: {xml_filename}")

except Exception as e:
    print(f"An error occurred: {e}")
