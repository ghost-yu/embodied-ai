ROM_DPATH=$1
TARGET_ROM_DPATH=$ROM_DPATH/roms

mkdir -p $TARGET_ROM_DPATH

# iterate through each of the zips in the directory
# for f in "$ROM_DPATH"/*.zip; do
#     #extract the name of the zip
#     name=$(basename "$f")
#     name="${name%.*}"

#     mkdir -p $TARGET_ROM_DPATH/"$name"
#     # extract the zip into the target rom dpath
#     unzip -o "$f" -d $TARGET_ROM_DPATH/"$name"
# done

# iterate through each of the newly extracted folders
for d in "$TARGET_ROM_DPATH"/*; do
    if [ -d "$d" ]; then
        # iterate through each of the zips in the directory
        for f in "$d"/*.zip; do
            # extract the zip into the target rom dpath
            unzip -o "$f" -d "$d"
            
            # delete the zip file
            rm "$f"
        done
    fi
done