function get_checker {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        checkmd5=md5
    else
        checkmd5=md5sum
    fi

    echo $checkmd5
}


function verify_1m {
    # From: curl -O http://files.grouplens.org/datasets/movielens/ml-1m.zip.md5
    hash=<(echo "MD5 (ml-1m.zip) = c4d9eecfca2ab87c1945afe126590906")
    local checkmd5=$(get_checker)
    if diff <($checkmd5 ml-1m.zip) $hash &> /dev/null
    then
        echo "PASSED"
    else
        echo "FAILED"
    fi
}

function verify_25m {
    # From: curl -O http://files.grouplens.org/datasets/movielens/ml-25m.zip.md5
    hash=<(echo "MD5 (ml-25m.zip) = 6b51fb2759a8657d3bfcbfc42b592ada")
    local checkmd5=$(get_checker)

    if diff <($checkmd5 ml-25m.zip) $hash &> /dev/null
    then
        echo "PASSED"
    else
        echo "FAILED"
    fi

}


if [[ $1 == "ml-1m" ]]
then
    verify_1m
else
    verify_25m
fi
