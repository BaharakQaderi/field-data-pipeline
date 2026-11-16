month = 6
day = 23

if 1 <= month <= 3:
    season = "winter"
elif 4 <= month <= 6:
    season = "spring"
elif 7 <= month <= 9:
    season = "summer"
else:
    season = "fall"

if month % 3 == 0 and day >= 21:
    if season == "winter":
        season = "spring"
    elif season == "spring":
        season = "summer"
    elif season == "summer":
        season = "fall"
    else:
        season = "winter"

print(season)
