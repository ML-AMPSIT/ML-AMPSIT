#!/bin/bash


# Read values from configAMPSIT.json
config_file="configAMPSIT.json"
totalsim=$(grep -Po '"totalsim": \K\d+' "$config_file")
parameter_names=$(grep -Po '(?<="parameter_names": )\[[^\]]+\]' "$config_file" | sed 's/"//g;s/\[//;s/\]//')
column=$(grep -Po '"vegtype": \K\d+' "$config_file")

echo "parameter_names: $parameter_names"
echo "column: $column"

IFS=',' read -r -a temp_array <<< "$parameter_names"
parameter_names=("${temp_array[@]}")
for param in "${parameter_names[@]}"; do
  echo "Parameter: $param"
done


# Create folder "folder_i" for each value from 1 to totalsim
folder=$(grep -Po '(?<="folder": ")[^"]+' "$config_file")
for ((i=1; i<=totalsim; i++)); do
  cp -r "$folder" "${folder}_$i"
  echo "Folder ${folder}_$i created"
done


comma=","

for ((i=1; i<=totalsim; i++)); do
  lineNo=$i
  line="$(head -n $i < X.txt | tail -1)" 
  IFS=' ' read -r -a paramsX <<< "$line"

  echo
  
  cd "${folder}_$i"
  echo "Entered in folder ${folder}_$i"



  # Loop to handle parameters
	for ((index=0; index<${#parameter_names[@]}; index++)); do
	  param="${parameter_names[index]}"
	  
	  # Get the index of the column corresponding to the parameter
	  column_index=$((index))
	  
	  # Read the value in the "param" column from X.txt
	  value=${paramsX[i,column_index]}
	
	  # Read falues from file MPTABLE.TBL
		
		mptable_values=$(grep -E "^${param}[[:space:]]*=" MPTABLE.TBL | awk 'NR==2')

	  
	  echo "mptable_values: $mptable_values"
	  
	  IFS=',' read -ra mptable_array <<< "$mptable_values"
	  
		# Function to separate elements of a string with the separator ","
		function join_with_comma {
		  local IFS=","
		  local arr=("$@")
		  echo "${arr[*]}"
		}

		  # Generate var1a from MPTABLE
		  var1a="$(join_with_comma "${mptable_array[@]:0:column}"),"
		  # Generate var1 from MPTABLE and X.txt
		  var1="$(join_with_comma "${mptable_array[@]:0:column-1}"), ${value}"
		  
		  echo "var1a = $var1a"
		  echo "var1 = $var1"



	  sed -i -e "s/$var1a/$var1,/g" MPTABLE.TBL
	  sed -i -e "s/$var1a/$var1,/g" MPTABLE.TBL

	  echo "Modified the MPTABLE.TBL file with the value of ${value}"
	done
  cd ..
  echo "Exited from folder folder_$i"
  echo "Done $i"
done
