DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

wget http://groups.csail.mit.edu/mug/natops/data/NATOPS24.zip
unzip ./NATOPS24.zip
rm NATOPS24.zip
