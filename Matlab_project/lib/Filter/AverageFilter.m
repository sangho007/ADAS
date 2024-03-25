
function avgValue = AverageFilter(newValue, windowSize)
    % newValue: 새로 들어온 데이터 값
    % windowSize: 이동 평균을 계산하기 위한 윈도우 크기
    % avgValue: 계산된 평균 값

    % 이전 데이터와 평균을 유지하기 위한 persistent 변수
    persistent dataWindow sumData numData

    % 첫 호출시 persistent 변수 초기화
    if isempty(dataWindow)
        dataWindow = zeros(1, windowSize); % 데이터 윈도우 초기화
        sumData = 0; % 데이터 합계 초기화
        numData = 0; % 윈도우 내 데이터 개수 초기화
    end

    % 새 데이터를 윈도우에 추가하고, 가장 오래된 데이터를 제거
    sumData = sumData - dataWindow(1) + newValue;
    dataWindow = [dataWindow(2:end), newValue];

    % 실제 데이터 개수 업데이트
    numData = min(windowSize, numData + 1);

    % 평균 계산
    avgValue = sumData / numData;
end
