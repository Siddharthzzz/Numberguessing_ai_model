#!/bin/bash
# Q2: Shell script task

read -p "Enter directory name: " dir

if [ ! -d "$dir" ]; then
  echo "Directory does not exist!"
  exit 1
fi

echo "Number of files: $(find "$dir" -type f | wc -l)"
echo "Number of subdirectories: $(find "$dir" -type d | wc -l)"

# Largest file
largest=$(find "$dir" -type f -printf "%s %p\n" | sort -nr | head -1)
echo "Largest file: $largest"

# Files modified in last 24 hrs
echo "First 10 lines of recently modified document files:"
find "$dir" -type f -mtime -1 -name "*.txt" | while read file; do
  echo "---- $file ----"
  head -10 "$file"
done
