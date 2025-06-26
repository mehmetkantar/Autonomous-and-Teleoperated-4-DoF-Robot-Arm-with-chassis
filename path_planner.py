import numpy as np
from math import sin, cos, pi, sqrt, atan2

class PathPlanner:
    """
    Class for generating various path types between two points in 3D space.
    Implements different motion profiles for robot arm path planning.
    """
    
    @staticmethod
    def generate_straight_line(start_point, end_point, num_points=50):
        """
        Generate a straight line path from start point to end point.
        
        Parameters:
        -----------
        start_point : tuple or list
            Starting point (x, y, z)
        end_point : tuple or list
            Ending point (x, y, z)
        num_points : int
            Number of points in the path
            
        Returns:
        --------
        list
            List of (x, y, z) points along the path
        """
        path = []
        
        # Convert to numpy arrays for easier calculation
        start = np.array(start_point)
        end = np.array(end_point)
        
        # Generate points along the line
        for i in range(num_points):
            t = i / (num_points - 1)  # Parameter from 0 to 1
            point = start + t * (end - start)
            path.append(tuple(point))
        
        return path
    
    @staticmethod
    def generate_half_circle(start_point, end_point, num_points=50, height_factor=0.5, direction="up"):
        """
        Generate a half-circle path from start point to end point.
        The path will form an arc with a height determined by height_factor.
        
        Parameters:
        -----------
        start_point : tuple or list
            Starting point (x, y, z)
        end_point : tuple or list
            Ending point (x, y, z)
        num_points : int
            Number of points in the path
        height_factor : float
            Factor determining how high the arc goes (0.5 = half the distance between points)
        direction : str
            Direction of the arc: "up" for above the line, "down" for below the line
            
        Returns:
        --------
        list
            List of (x, y, z) points along the path
        """
        path = []
        
        # Convert to numpy arrays
        start = np.array(start_point)
        end = np.array(end_point)
        
        # Find midpoint
        midpoint = (start + end) / 2
        
        # Calculate the direction vector from start to end
        direction_vector = end - start
        distance = np.linalg.norm(direction_vector)
        
        # Find a perpendicular vector in the vertical direction
        if abs(direction_vector[2]) < 1e-6 and abs(direction_vector[0]) < 1e-6 and abs(direction_vector[1]) < 1e-6:
            # If direction is zero (start = end), just return the point
            return [start_point] * num_points
        
        # Default: Lift in the z-direction
        perpendicular = np.array([0, 0, 1])
        
        # If direction is primarily in z, lift in the y-direction
        if abs(direction_vector[2]) > abs(direction_vector[0]) and abs(direction_vector[2]) > abs(direction_vector[1]):
            perpendicular = np.array([0, 1, 0])
        
        # Apply direction (up or down)
        if direction.lower() == "down":
            perpendicular = -perpendicular
            
        # Calculate the peak of the arc
        peak = midpoint + perpendicular * distance * height_factor
        
        # Generate points along the half-circle
        for i in range(num_points):
            t = i / (num_points - 1)  # Parameter from 0 to 1
            angle = t * pi  # Angle from 0 to pi for half-circle
            
            # Parametric equation of half-circle
            # Linear interpolation base (straight line between points)
            point = start * (1 - t) + end * t
            
            # Add the arc height (maximum at the middle, zero at the ends)
            arc_height = perpendicular * distance * height_factor * sin(angle)
            
            # Adjust the point to follow the arc
            point = point + arc_height
            
            path.append(tuple(point))
        
        return path
    
    @staticmethod
    def generate_half_figure8(start_point, end_point, num_points=50, height_factor=0.5):
        """
        Generate a half-figure-8 path from start point to end point.
        The path will go up in a half circle then down in another half circle.
        
        Parameters:
        -----------
        start_point : tuple or list
            Starting point (x, y, z)
        end_point : tuple or list
            Ending point (x, y, z)
        num_points : int
            Number of points in the path
        height_factor : float
            Factor determining how high/low the arcs go
            
        Returns:
        --------
        list
            List of (x, y, z) points along the path
        """
        path = []
        
        # Convert to numpy arrays
        start = np.array(start_point)
        end = np.array(end_point)
        
        # Find quarter points
        first_quarter = start + (end - start) * 0.25
        midpoint = (start + end) / 2
        third_quarter = start + (end - start) * 0.75
        
        # Calculate the direction vector from start to end
        direction_vector = end - start
        distance = np.linalg.norm(direction_vector)
        
        # If start and end are identical, just return the point
        if distance < 1e-6:
            return [start_point] * num_points
        
        # Find perpendicular vectors
        # Primary lifting direction (z-axis by default)
        up_vector = np.array([0, 0, 1])
        
        # If direction is primarily in z, use y-axis for lifting
        if abs(direction_vector[2]) > abs(direction_vector[0]) and abs(direction_vector[2]) > abs(direction_vector[1]):
            up_vector = np.array([0, 1, 0])
        
        # Generate points along the half-figure-8
        for i in range(num_points):
            t = i / (num_points - 1)  # Parameter from 0 to 1
            
            # Linear interpolation base (straight line)
            point = start * (1 - t) + end * t
            
            # Calculate vertical displacement
            if t < 0.5:
                # First half: goes up
                t_local = t * 2  # Scale to 0-1 for first half
                height = height_factor * distance * sin(t_local * pi)
                # Add upward arc
                point = point + up_vector * height
            else:
                # Second half: goes down
                t_local = (t - 0.5) * 2  # Scale to 0-1 for second half
                height = height_factor * distance * sin(t_local * pi)
                # Add downward arc
                point = point - up_vector * height
            
            path.append(tuple(point))
        
        return path
    
    @staticmethod
    def generate_path(path_type, start_point, end_point, num_points=50, height_factor=0.5):
        """
        Generate a path based on the specified path type.
        
        Parameters:
        -----------
        path_type : str
            Type of path: 'straight', 'half_circle_up', 'half_circle_down', 'half_figure8'
        start_point : tuple or list
            Starting point (x, y, z)
        end_point : tuple or list
            Ending point (x, y, z)
        num_points : int
            Number of points in the path
        height_factor : float
            Factor determining how high the arcs go (for non-straight paths)
            
        Returns:
        --------
        list
            List of (x, y, z) points along the path
        """
        if path_type == 'straight':
            return PathPlanner.generate_straight_line(start_point, end_point, num_points)
        elif path_type == 'half_circle_up':
            return PathPlanner.generate_half_circle(start_point, end_point, num_points, height_factor, "up")
        elif path_type == 'half_circle_down':
            return PathPlanner.generate_half_circle(start_point, end_point, num_points, height_factor, "down")
        elif path_type == 'half_figure8':
            return PathPlanner.generate_half_figure8(start_point, end_point, num_points, height_factor)
        else:
            raise ValueError(f"Unknown path type: {path_type}")


# Test the path planner if run directly
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Define test points
    start = (0, 0, 10)
    end = (20, 20, 10)
    
    # Generate paths with different path types
    straight_path = PathPlanner.generate_path('straight', start, end)
    half_circle_up_path = PathPlanner.generate_path('half_circle_up', start, end)
    half_circle_down_path = PathPlanner.generate_path('half_circle_down', start, end)
    half_figure8_path = PathPlanner.generate_path('half_figure8', start, end)
    
    # Plot the paths
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates for each path
    straight_x, straight_y, straight_z = zip(*straight_path)
    half_circle_up_x, half_circle_up_y, half_circle_up_z = zip(*half_circle_up_path)
    half_circle_down_x, half_circle_down_y, half_circle_down_z = zip(*half_circle_down_path)
    half_figure8_x, half_figure8_y, half_figure8_z = zip(*half_figure8_path)
    
    # Plot the paths
    ax.plot(straight_x, straight_y, straight_z, 'r-', label='Straight Line')
    ax.plot(half_circle_up_x, half_circle_up_y, half_circle_up_z, 'g-', label='Half Circle Up')
    ax.plot(half_circle_down_x, half_circle_down_y, half_circle_down_z, 'y-', label='Half Circle Down')
    ax.plot(half_figure8_x, half_figure8_y, half_figure8_z, 'b-', label='Half Figure-8')
    
    # Plot start and end points
    ax.scatter([start[0]], [start[1]], [start[2]], color='green', s=100, label='Start')
    ax.scatter([end[0]], [end[1]], [end[2]], color='red', s=100, label='End')
    
    # Set labels and title
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title('Robot Arm Path Trajectories')
    
    # Set a good viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add legend
    ax.legend()
    
    # Show the plot
    plt.savefig('path_trajectories.png')
    plt.show()
