from pxr import Usd, UsdPhysics

# 修改为你的 USD 文件的绝对路径或相对路径
usd_file_path = "USD/COD-2026RoboMaster-Balance.usd"

def print_joints(usd_path):
    try:
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"Error: Could not open file {usd_path}")
            return

        print(f"\n--- Analyzing: {usd_path} ---")
        print(f"{'Joint Name':<30} | {'Type':<20} | {'Prim Path'}")
        print("-" * 80)

        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                # 获取关节名称（这是你在 config 中需要的）
                joint_name = prim.GetName()
                
                # 获取关节类型
                type_name = "Unknown"
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    type_name = "Revolute (旋转)"
                elif prim.IsA(UsdPhysics.PrismaticJoint):
                    type_name = "Prismatic (平移)"
                elif prim.IsA(UsdPhysics.FixedJoint):
                    type_name = "Fixed (固定)"
                
                print(f"{joint_name:<30} | {type_name:<20} | {prim.GetPath()}")
        print("-" * 80)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print_joints(usd_file_path)