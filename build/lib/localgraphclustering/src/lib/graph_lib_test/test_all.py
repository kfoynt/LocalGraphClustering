from subprocess import call

test_results = []
size = 0
result = call(["./aclpagerank"])
test_results = test_results + [("aclpagerank", result)]
size = size + 1
result = call(["./sweepcut"])
test_results = test_results + [("sweepcut", result)]
size = size + 1
result = call(["./MQI"])
test_results = test_results + [("MQI", result)]
size = size + 1
result = call(["./ppr_path"])
test_results = test_results + [("ppr_path", result)]
size = size + 1
result = call(["./aclpagerank_weighted"])
test_results = test_results + [("aclpagerank_weighted", result)]
size = size + 1
print("\n\ntest summary:")
for i in range(size):
    result = test_results[i]
    if result[1] == 0:
        print('{0:30s} {1:10s}'.format(result[0],"pass"))
    else:
        print('{0:30s} {1:10s}'.format(result[0],"fail"))

