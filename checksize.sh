#!/bin/bash

name="foldersim_"

file="wrfout_d01_2015-03-20_*"

minimumsize=16633374696 

for i in {1..200}
do
cd $name$i
if [ -f $file ]; then
size=$(wc -c $file | awk '{print $1}')

if (( $size == 16633374696  )); then
echo "size equal to $minimumsize ($size) in $i"
mv $file wrfout_d01_2015-03-20_12:00:00_$i
mv wrfout_d01_2015-03-20_12:00:00_$i ../out/
echo "$i moved"
cd ..
else
echo "size under $minimumsize ($size) in $i"
cd ..
fi

else
echo "file does not exist in $i"
cd ..
fi
done



