sed -i 's/[^,]*,//' x.csv 
sed -i 's/[^,]*,//' y.csv 
sed -i '1iRefId,IsBadBuy' output.csv 
