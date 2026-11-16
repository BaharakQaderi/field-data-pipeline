function table = read_ipok(filePath)
    % read_ipok - Reads and processes a CSV file containing time-based data.
    %
    % INPUT:
    %   filePath - String, path to the CSV file.
    %
    % OUTPUT:
    %   table - Table containing filtered data from the year 2025.

    % Read the CSV file into a table
    table = readtable(filePath, 'Delimiter', ',');

    % Convert 'FormattedUtcTime' column to datetime format
    table.FormattedUtcTime = datetime(table.FormattedUtcTime, ...
        'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSSSSS ''UTC''', 'TimeZone', 'UTC');

    table.FormattedUtcTime.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSSXXX';

    % Filter rows for the year 2025
    table = table(year(table.FormattedUtcTime) == 2025, :);
end