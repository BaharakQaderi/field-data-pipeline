function cleanedData = clean(filteredData)
	% Fill missing data
	cleanedData = fillmissing(filteredData,"linear", ...
	    "DataVariables",["Latitude","Longitude","Altitude"]);
end