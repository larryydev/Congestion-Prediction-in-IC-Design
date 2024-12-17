import os

class DataLoader:

    @staticmethod
    def get_all_file_paths(root_dir='./dataset'):
        all_files = {}
        file_counts = {}
        for root, _, files in os.walk(root_dir):
            dir_name = os.path.basename(root)
            current_files = []
            count = 0
            for file in files:
                if file != ".DS_Store":
                    file_path = os.path.join(root, file)
                    current_files.append(file_path)
                    count += 1
            if current_files:
                all_files[dir_name] = current_files
                file_counts[dir_name] = count
        return all_files, file_counts['cell_density']

    @staticmethod
    def get_congestion():
        return DataLoader.get_all_file_paths(root_dir='./dataset/congestion')

    @staticmethod
    def get_cell_density():
        return DataLoader.get_all_file_paths(root_dir='./dataset/cell_density')

    @staticmethod
    def get_macro_region():
        return DataLoader.get_all_file_paths(root_dir='./dataset/macro_region')
    
    @staticmethod
    def get_RUDY():
        return DataLoader.get_all_file_paths(root_dir='./dataset/RUDY')
    

if __name__ == '__main__':
    d, c = DataLoader.get_all_file_paths()

