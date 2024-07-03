import sys
from math import ceil

def split_file(input_filename, n):
    # Open the input file and read lines
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    # Calculate the number of lines per file
    total_lines = len(lines)
    lines_per_file = ceil(total_lines / n)

    # Split lines into n parts and write to new files
    for i in range(n):
        start_index = i * lines_per_file
        end_index = min((i + 1) * lines_per_file, total_lines)
        part_lines = lines[start_index:end_index]
        output_filename = f'{i}'
        
        with open(output_filename, 'w') as output_file:
            output_file.writelines(part_lines)
        print(f'Created {output_filename} with {len(part_lines)} lines.')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python split_file.py <input_file> <number_of_files>")
        sys.exit(1)

    input_file = sys.argv[1]
    number_of_files = int(sys.argv[2])

    split_file(input_file, number_of_files)
