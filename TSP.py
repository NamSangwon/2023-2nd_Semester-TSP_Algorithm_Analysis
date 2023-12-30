import csv
import json
import urllib
import requests
import sys
from urllib.request import urlopen
import time
import random

INF = 999999

# 데이터 이동 연산 횟수 & 데이터 비교 연산 횟수
BRUTE_MOVE = 0
BRUTE_COMP = 0
GREEDY_MOVE = 0
GREEDY_COMP = 0
DP_MOVE = 0
DP_COMP = 0

N = int(input("총 방문하고 싶은 관광지 개수를 입력하시오. >> "))
tour_list = []
recommend_tour_list = []
choose_subcategory = {}

def read_file():
    max_search_cnt = 0
    min_search_cnt = 0
    subcategory_list = []
    with open("tour_list.csv", "r") as f:
        filter_name = []
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "순위": 
                filter_name = row
                continue

            tour = dict(zip(filter_name, row)) # array to dictionary
            if tour['순위'] == "1": max_search_cnt = tour['검색건수'] # 첫번째 행 값이 최대값 
            min_search_cnt = tour['검색건수'] # 마지막 행 값이 최소값
            if tour['소분류 카테고리'] not in subcategory_list:
                subcategory_list.append(tour['소분류 카테고리'])
            tour_list.append(tour)

    return subcategory_list, int(max_search_cnt), int(min_search_cnt)

def select_subcategory(subcategory_list):
    # 입력 값에 따른 관광지 추천
    total = N
    wish_subcategory = []
    while(total > 0):
        category = input("\n방문하고 싶은 관광지의 카테고리를 입력하시오. >> ")
        cnt = int(input("해당 카테고리 관광지의 원하는 최대 방문 개수를 입력하시오. >> "))
        total -= cnt
        if category not in subcategory_list: continue
        choose_subcategory[category] = cnt
        wish_subcategory.append(category)
        
    return wish_subcategory

def select_category(subcategory):
    wish_category = []
    for tour in tour_list:
        if tour['소분류 카테고리'] not in subcategory: continue
        if tour['중분류 카테고리'] in wish_category: continue
        wish_category.append(tour['중분류 카테고리'])
    return wish_category

# 주소에 geocoding 적용하는 함수를 작성.
def get_location(loc) :
    client_id = "API-ID"
    client_secret = "API-Key"
    url = f"https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query=" \
    			+ urllib.parse.quote(loc)
    
    # 주소 변환
    request = urllib.request.Request(url)
    request.add_header('X-NCP-APIGW-API-KEY-ID', client_id)
    request.add_header('X-NCP-APIGW-API-KEY', client_secret)
    
    response = urlopen(request)
    res = response.getcode()
    
    if (res == 200) : # 응답이 정상적으로 완료되면 200을 return한다
        response_body = response.read().decode('utf-8')
        response_body = json.loads(response_body)
        # print(response_body) # 읽어 온 json 값 출력
        # 주소가 존재할 경우 total count == 1이 반환됨.
        if response_body['meta']['totalCount'] == 1 : 
        	# 위도, 경도 좌표를 받아와서 return해 줌.
            lat = response_body['addresses'][0]['y']
            lon = response_body['addresses'][0]['x']
            return (lon, lat)
        else :
            print('location not exist')
        
    else :
        print('ERROR')

# 출발지 -> 도착지 길찾기 (소요 시간만을 반환)
def get_timeRequired(start, goal) : 
    try:
        client_id = "API-ID"
        client_secret = "API-Key"

        url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
        params = {
            "start": f"{start[0]},{start[1]}",
            "goal": f"{goal[0]},{goal[1]}"
        }
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
        }
        response = requests.get(url, params=params, headers=headers)
        res_json = response.json()

        return int(res_json['route']['traoptimal'][0]['summary']['duration'] / 60000)

    except requests.HTTPError as e:
        print(e)

# data = [관광지명, 점수] (분류 기준 = 관광지 점수)
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    lesser_arr, equal_arr, greater_arr = [], [], []
    for num in arr:
        if num['점수'] < pivot['점수']:
            lesser_arr.append(num)
        elif num['점수'] > pivot['점수']:
            greater_arr.append(num)
        else:
            equal_arr.append(num)
    return quick_sort(greater_arr) + equal_arr + quick_sort(lesser_arr)

# ------------------------------brute-force----------------------------------------------------------
# calculating distance of the cities
def calculate_total_time(route_times_list, order):
    total_sum = 0

    for i in range(N-1):
        total_sum += route_times_list[order[i]][order[i+1]]

    return total_sum
    
# 순열 계산
def permute(arr):
    global BRUTE_COMP
    global BRUTE_MOVE

    result = [arr[:]]
    c = [0] * len(arr)
    i = 0
    while i < len(arr):
        BRUTE_COMP += 1 # count data comp
        if c[i] < i:
            BRUTE_COMP += 1 # count data comp
            if i % 2 == 0:
                arr[0], arr[i] = arr[i], arr[0]
            else:
                arr[c[i]], arr[i] = arr[i], arr[c[i]]
            BRUTE_MOVE += 2 # count data move (swap element in order)
            result.append(arr[:])
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1
    return result

def brute_force_search(route_times_list):
    global BRUTE_COMP
    global BRUTE_MOVE

    orders = permute(list(range(N)))

    # 경로 중복 검사 (A-B-C-D == D-C-B-A 제거)
    for permutation in orders:
        BRUTE_COMP += 1 # count data compare 
        if list(reversed(permutation)) in orders:
            BRUTE_MOVE += N # count data move (delete array element)
            orders.remove(list(reversed(permutation)))

    min_total_time = sys.maxsize
    min_route = []

    for order in orders:
        total_time = calculate_total_time(route_times_list, order)
        BRUTE_COMP += 1 # count data compare 
        if min_total_time > total_time:
            BRUTE_MOVE += 2 # count data move twice
            min_total_time = total_time
            min_route = order

    return min_total_time, min_route
#-----------------------------------------------------------------------------------------------------
#-----------------------dynamic programming-----------------------------------------------------------
def subsets_containing_k_vertices(n, k):
    global DP_COMP
    global DP_MOVE

    S = []
    for A in range(2 ** (n-1)):
        DP_COMP += 1 # count data comp
        if bin(A).count("1") == k:
            DP_MOVE += 1 # count data move
            S.append(A)
    return S

def not_in_A(n, A):
    global DP_COMP
    global DP_MOVE

    S = []
    for i in range(n - 1):
        DP_COMP += 1 # count data comp
        if A & (1 << i) == 0:
            DP_MOVE += 1 # count data move
            S.append(2 + i)
    return S

def in_A(n, A):
    global DP_COMP
    global DP_MOVE

    S = []
    for i in range(n - 1):
        DP_COMP += 1 # count data comp
        if A & (1 << i) != 0:
            DP_MOVE += 1 # count data move
            S.append(2 + i)
    return S

def diff(A, j):
    return A & ~(1 << (j - 2))

def minimum(n, i, A, W, D):
    global DP_COMP
    global DP_MOVE
    global INF
    minvalue, minj = INF, 0

    for j in in_A(n, A):
        DP_MOVE += 1 # count data move
        value = W[i][j] + D[j][diff(A, j)]
        DP_COMP += 1 # count data comp
        if value < minvalue:
            DP_MOVE += 2 # count data move
            minvalue, minj = value, j

    return minvalue, minj

def expand_matrix(matrix): #행렬 확장 함수; n*n행렬에 원소가 모두 0인 새로운 1행과 1열 추가 삽입
    global DP_COMP
    global DP_MOVE
    global INF
    
    n = len(matrix)
    # 원소가 모두 0인 크기 n+1의 새로운 행을 준비합니다.
    zero_row = [0] * (n+1)
    # 원소가 모두 INF인 크기 n+1의 새로운 행을 준비합니다.
    INF_row = [INF] * (n+2)
    
    # 원래의 행렬에 새로운 열을 추가합니다.
    for i in range(n):
        DP_MOVE += 1 # count data move (add element)
        matrix[i].insert(0, 0)

    # 원래의 행렬에 새로운 행을 추가합니다.
    DP_MOVE += N + 1 # count data move (elements in zero_row array)
    matrix.insert(0, zero_row)

    # 원래의 행렬에 새로운 열을 추가합니다.(dummy)
    for i in range(n + 1):
        DP_MOVE += 1 # count data move (add element)
        matrix[i].insert(0, INF)

    # 원래의 행렬에 새로운 행을 추가합니다.(dummy)
    DP_MOVE += N + 2 # count data move (elements in INF_row array)
    matrix.insert(0, INF_row)

    #route_times_list (x,x) = infinite를 (x,x) = 0으로 수정
    for i in range(1,n + 2):
        DP_MOVE += 1 # count data move 
        matrix[i][i] = 0

    return matrix


def travel(n, W):
    global DP_COMP
    global DP_MOVE
    global INF

    D = [[INF] * (2**(n-1)) for _ in range(n + 1)]
    P = [[0] * (2**(n-1)) for _ in range(n + 1)]

    for i in range(2, n + 1):
        DP_MOVE += 1 # count data move
        D[i][0] = W[i][1]
        
    for k in range(1, n - 1):
        for A in subsets_containing_k_vertices(n, k):
            for i in not_in_A(n, A):
                DP_MOVE += 2 # count data move
                D[i][A], P[i][A] = minimum(n, i, A, W, D)
                
    A = 2 ** (n - 1) - 1 # A = V - {v1}
    DP_MOVE += 2 # count data move
    D[1][A], P[1][A] = minimum(n, 1, A, W, D)

    return D[1][A], D, P

def DP(route_times_list, N): # N: 방문하고싶은 관광지 수
    global DP_COMP
    global DP_MOVE

    newMatrix = expand_matrix(route_times_list)
    min_total_time, D, P = travel(N + 1, newMatrix)
    #잘 나오나 테스트---------------------------------
    # print("D = ")
    # for i in range(1, N + 1):
    #     print(D[i])
    # print("P = ")
    # for i in range(1, N + 1):
    #     print(P[i])
    #-------------------------------------------------
    A = 2 ** N - 1
    min_route = []
    nextNode = 1
    while(A != 0):
        DP_COMP += 1 # count data comp (in while statement) 
        #nextIndex = nextNode - 1 #노드 넘버링과 행렬 인덱스 넘버링 차이 조정
        nextNode = P[nextNode][A]
        A = diff(A, nextNode)
        DP_MOVE += 1 # count data move
        min_route.append(nextNode - 2) ## index조정
    
    return min_total_time, min_route
#-------------------------------------------------------------------
#----------------------Greedy---------------------------------------
def greedy(route_times_list, N):
    global GREEDY_COMP
    global GREEDY_MOVE

    min_total_time = float('inf')
    min_route = [] # 관광 순서 리스트 

    for i in range(N): # 모든 노드를 시작으로 지정
        visited = [False] * N  # 방문한 관광지 리스트
        current_node = i
        route = [current_node]
        visited[current_node] = True  # 시작 노드를 방문했다고 표시

        total_time = 0  # 총 이동 시간
        
        for _ in range(N - 1):  # 모든 노드 방문
            min_time = float('inf')  
            for j in range(N):  
                if visited[j]:  
                    continue
                GREEDY_COMP += 1
                if route_times_list[current_node][j] < min_time: 
                    min_time = route_times_list[current_node][j] 
                    next_node = j  

            total_time += min_time
            GREEDY_MOVE += 1
            route.append(next_node)
            visited[next_node] = True 
            current_node = next_node
            
        GREEDY_COMP += 1 # count data comp
        if min_total_time > total_time:
            GREEDY_MOVE += 1 + N # count data move (total time & elements in array)
            min_total_time = total_time
            min_route = route.copy()

    return min_total_time, min_route
#-------------------------------------------------------------------


def tour_score(wish_category, wish_subcategory, max, min):
    score = 0
    for tour in tour_list:
        if tour['소분류 카테고리'] in wish_subcategory: score += 1
        if tour['중분류 카테고리'] in wish_category: score += 1
        score += (int(tour['검색건수']) - min) / (max - min)
        tour['점수'] = score
        score = 0

def recommend_tour():
    recommend_tour = []
    total = 0
    for tour in tour_list:
        if total == N: break

        if choose_subcategory[tour['소분류 카테고리']] > 0:
            recommend_tour.append(tour)
            choose_subcategory[tour['소분류 카테고리']] -= 1
            total += 1
        
        if tour['소분류 카테고리'] not in choose_subcategory:
            recommend_tour.append(tour)
            total += 1
        
    return recommend_tour

if __name__ == '__main__':

    # 관광지 데이터 읽기
    subcategory_list, max_search_cnt, min_search_cnt = read_file()

    # 카테고리 값 출력
    print("\n\n[ 소분류 카테고리 ]")
    for category in subcategory_list:
        print(category)

    # 추천 관광지 카테고리 분류
    wish_subcategory = select_subcategory(subcategory_list)
    wish_category = select_category(wish_subcategory)

    # 관광지 점수 매기기 (= 중분류 카테고리 + 소분류 카테고리 + 검색건수(정규화))
    tour_score(wish_category, wish_subcategory, max_search_cnt, min_search_cnt)

    # 관광지 점수 별로 정렬
    tour_list = quick_sort(tour_list)
    
    # 관광지 추천
    recommend_tour_list = recommend_tour()
    
    print("\n[[ 추천 관광지 ]]")
    for tour in recommend_tour_list:
        print(tour['관광지명'], tour['점수'])

    route_times_list = []
    
    ########################### API 사용 ####################################
    # 추천 관광지 주소를 위도 경도로 변환 후 관광지 정보에 append (Geocoding API)
    for tourist in recommend_tour_list:
        loc = get_location(tourist['도로명주소'] + " " + tourist['관광지명'])
        tourist['위도'] = loc[0]
        tourist['경도'] = loc[1]
    
    # 관광지 간의 소요 시간 계산 (분 단위) (Direction5 API)
    for i in range(N):
        drive_time = []
        # for destination in recommend_tour_list:
        for j in range(N):
            if i >= j: 
                drive_time.append(0)
            else:
                arrival = (recommend_tour_list[i]['위도'], recommend_tour_list[i]['경도'])
                destination = (recommend_tour_list[j]['위도'], recommend_tour_list[j]['경도'])
                result = get_timeRequired(arrival, destination)
                drive_time.append(result)
        route_times_list.append(drive_time)

    ########################### API 사용안하고 테스팅 ############################
    # route_times_list = []

    # for i in range(N):
    #     lst = []
    #     for j in range(N):
    #         if i == j: 
    #             lst.append(0) 
    #             continue
    #         lst.append(random.randrange(10,150))
    #     route_times_list.append(lst)

    # (a -> b) == (b -> a) 이므로 값 복사
    for i in range(N):
        for j in range(N):
            if i <= j: route_times_list[j][i] = route_times_list[i][j]

    # 각 관광지 간의 소요 시간 출력
    print("\n[[ 각 관광지 별 이동 소요 시간 ]]")
    for route_times in route_times_list:
        print(route_times)

    # Brute-Force
    brute_force_start_time = time.time()
    shortest_total_time, shortest_route = brute_force_search(route_times_list)
    #output - brute-force
    print("\n[[ 최단 소요 시간 경로 출력 (Brute-Force) ]]")
    print("shortest_total_time : ", shortest_total_time, "분")
    print("shortest_route : ", end = '')
    for i in range(len(shortest_route) - 1):
        cur = shortest_route[i]
        next = shortest_route[i+1]
        print(recommend_tour_list[cur]['관광지명'], end=' ')
        if i != len(shortest_route): print("-(", route_times_list[cur][next], ")->", end=' ')
    print(recommend_tour_list[shortest_route[-1]]['관광지명'])
    brute_force_end_time = time.time()

    # Greedy Algoritm
    greedy_start_time = time.time()
    hortest_total_time_greedy, shortest_route_greedy = greedy(route_times_list, N)
    #output - greedy
    print("\n[[ 최단 소요 시간 경로 출력 (Greedy) ]]")
    print("shortest_total_time : ", hortest_total_time_greedy, "분")
    print("shortest_route : ", end = '')
    for i in range(len(shortest_route_greedy) - 1):
        cur = shortest_route_greedy[i]
        next = shortest_route_greedy[i+1]
        print(recommend_tour_list[cur]['관광지명'], end=' ')
        if i != len(shortest_route_greedy): print("-(", route_times_list[cur][next], ")->", end=' ')
    print(recommend_tour_list[shortest_route_greedy[-1]]['관광지명'])
    greedy_end_time = time.time()

    # Dynamic programming
    dp_start_time = time.time()
    shortest_total_time_test, shortest_route_test = DP(route_times_list, N)
    #output - Dynamic programming
    print("\n[[ 최단 소요 시간 경로 출력 (DP) ]]")
    print("shortest_total_time : ", shortest_total_time_test, "분")
    print("shortest_route : ", end = '')
    for i in range(len(shortest_route_test) - 1):
        cur = shortest_route_test[i]
        next = shortest_route_test[i+1]
        print(recommend_tour_list[cur]['관광지명'], end=' ')
        if i != len(shortest_route_test): print("-(", route_times_list[cur+2][next+2], ")->", end=' ')
    print(recommend_tour_list[shortest_route_test[-1]]['관광지명'])
    dp_end_time = time.time()

    print("\n[[ 알고리즘 비교 분석 ]]")
    print(f"<Brute-Force>\n데이터 비교 연산 : {BRUTE_COMP}| 데이터 이동 연산 : {BRUTE_MOVE} | 알고리즘 수행 소요 시간 : {brute_force_end_time - brute_force_start_time:.5f}")
    print(f"<Greedy>\n데이터 비교 연산 : {GREEDY_COMP} | 데이터 이동 연산 : {GREEDY_MOVE} | 알고리즘 수행 소요 시간 : {greedy_end_time - greedy_start_time:.5f}")
    print(f"<DP>\n데이터 비교 연산 : {DP_MOVE} | 데이터 이동 연산 : {DP_MOVE} | 알고리즘 수행 소요 시간 : {dp_end_time - dp_start_time:.5f}")