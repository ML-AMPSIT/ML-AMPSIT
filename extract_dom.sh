
for i in {1..100}
do
sim=wrfout_d01_2015-03-20_18\:00\:00_
newname=wrfout_d01_2015-03-20_18_00_00_
echo ${newname}
ncks -v LU_INDEX,PH,PHB,Q2,T2,TH2,PSFC,Times,U_MEAN,WSPV_MEAN,V_MEAN,U_10_MEAN,V_10_MEAN,T_2_MEAN,TH_MEAN,HGT,TSK,HFX,LH ${sim}$i ${newname}$i
ncks -O -F -d south_north,76,126 ${newname}$i ${newname}$i
ncks -O -F -d west_east,76,126 ${newname}$i ${newname}$i
#ncks -O -F -d Time,1,24 ${newname}$i ${newname}$i
done
