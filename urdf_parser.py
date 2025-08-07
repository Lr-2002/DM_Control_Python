#!/usr/bin/env python3
"""
URDF Parser for extracting DH parameters from robot URDF file
"""

import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

class URDFParser:
    def __init__(self, urdf_path):
        """
        Initialize URDF parser
        
        Args:
            urdf_path: Path to URDF file
        """
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()
        
    def extract_joint_info(self):
        """
        Extract joint information from URDF
        
        Returns:
            dict: Joint information including origins and axes
        """
        joints = {}
        
        for joint in self.root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            
            # Extract origin (position and orientation)
            origin = joint.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0').split()
                rpy = origin.get('rpy', '0 0 0').split()
                position = [float(x) for x in xyz]
                orientation = [float(x) for x in rpy]
            else:
                position = [0, 0, 0]
                orientation = [0, 0, 0]
            
            # Extract axis
            axis_elem = joint.find('axis')
            if axis_elem is not None:
                axis = [float(x) for x in axis_elem.get('xyz', '0 0 1').split()]
            else:
                axis = [0, 0, 1]
            
            # Extract parent and child links
            parent = joint.find('parent')
            child = joint.find('child')
            parent_link = parent.get('link') if parent is not None else None
            child_link = child.get('link') if child is not None else None
            
            joints[joint_name] = {
                'type': joint_type,
                'position': position,
                'orientation': orientation,
                'axis': axis,
                'parent': parent_link,
                'child': child_link
            }
        
        return joints
    
    def convert_to_dh_parameters(self):
        """
        Convert URDF joint information to DH parameters
        
        Returns:
            dict: DH parameters (a, d, alpha, theta_offset)
        """
        joints = self.extract_joint_info()
        
        # Sort joints by name (j1, j2, j3, j4, j5)
        joint_names = sorted([name for name in joints.keys() if name.startswith('j')])
        
        dh_params = {
            'a': [],      # link lengths
            'd': [],      # link offsets  
            'alpha': [],  # link twists
            'theta_offset': []  # joint angle offsets
        }
        
        print("=== URDF Joint Information ===")
        for joint_name in joint_names:
            joint = joints[joint_name]
            print(f"\n{joint_name}:")
            print(f"  Position: {joint['position']}")
            print(f"  Orientation (RPY): {joint['orientation']}")
            print(f"  Axis: {joint['axis']}")
            print(f"  Parent: {joint['parent']} -> Child: {joint['child']}")
            
            # Extract DH parameters from joint transform
            pos = np.array(joint['position'])
            rpy = np.array(joint['orientation'])
            
            # Link length (a): distance along x-axis
            a = pos[0] if abs(pos[0]) > 1e-6 else 0
            
            # Link offset (d): distance along z-axis  
            d = pos[2] if abs(pos[2]) > 1e-6 else 0
            
            # Link twist (alpha): rotation about x-axis
            alpha = rpy[0] if abs(rpy[0]) > 1e-6 else 0
            
            # Joint angle offset (theta): initial rotation about z-axis
            theta_offset = rpy[2] if abs(rpy[2]) > 1e-6 else 0
            
            dh_params['a'].append(a)
            dh_params['d'].append(d) 
            dh_params['alpha'].append(alpha)
            dh_params['theta_offset'].append(theta_offset)
            
            print(f"  DH Parameters: a={a:.4f}, d={d:.4f}, alpha={alpha:.4f}, theta_offset={theta_offset:.4f}")
        
        return dh_params
    
    def get_link_masses(self):
        """
        Extract link mass information from URDF
        
        Returns:
            dict: Link masses and inertial properties
        """
        links = {}
        
        for link in self.root.findall('link'):
            link_name = link.get('name')
            
            inertial = link.find('inertial')
            if inertial is not None:
                # Mass
                mass_elem = inertial.find('mass')
                mass = float(mass_elem.get('value')) if mass_elem is not None else 0.0
                
                # Center of mass
                origin = inertial.find('origin')
                if origin is not None:
                    com_xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
                    com_rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
                else:
                    com_xyz = [0, 0, 0]
                    com_rpy = [0, 0, 0]
                
                # Inertia tensor
                inertia_elem = inertial.find('inertia')
                if inertia_elem is not None:
                    inertia = {
                        'ixx': float(inertia_elem.get('ixx', 0)),
                        'ixy': float(inertia_elem.get('ixy', 0)),
                        'ixz': float(inertia_elem.get('ixz', 0)),
                        'iyy': float(inertia_elem.get('iyy', 0)),
                        'iyz': float(inertia_elem.get('iyz', 0)),
                        'izz': float(inertia_elem.get('izz', 0))
                    }
                else:
                    inertia = {}
                
                links[link_name] = {
                    'mass': mass,
                    'com_position': com_xyz,
                    'com_orientation': com_rpy,
                    'inertia': inertia
                }
        
        return links

def main():
    """Test the URDF parser"""
    urdf_path = '/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_urdf/urdf/ic1.1.2.urdf'
    
    parser = URDFParser(urdf_path)
    
    # Extract DH parameters
    dh_params = parser.convert_to_dh_parameters()
    
    print("\n=== Final DH Parameters ===")
    print(f"a (link lengths): {dh_params['a']}")
    print(f"d (link offsets): {dh_params['d']}")  
    print(f"alpha (link twists): {dh_params['alpha']}")
    print(f"theta_offset (joint offsets): {dh_params['theta_offset']}")
    
    # Extract link masses
    links = parser.get_link_masses()
    
    print("\n=== Link Mass Information ===")
    for link_name, link_info in links.items():
        print(f"{link_name}: mass={link_info['mass']:.4f} kg, COM={link_info['com_position']}")

if __name__ == "__main__":
    main()
