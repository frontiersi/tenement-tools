from ._bapy import *

import math
import sys
import os

def getLocalDatasets():
	return getLocalDatasets_impl()

def getDataCollections(dataset):
	return getDataCollections_impl(dataset)

def getGeographyLevels(dataset):
	return getGeographyLevels_impl(dataset)

def getStandardGeographies(dataset, level = None, query = None, sublevel = None, subquery = None):
    if (level != None and query != None and sublevel != None and subquery != None):
        return getSubgeographies_impl(dataset, level, query, sublevel, subquery)
    elif (level != None and query != None and sublevel != None):
        return getSubgeographies_impl(dataset, level, query, sublevel)
    elif (level != None and query != None):
        return getSubgeographies_impl(dataset, level, query)
    elif (level != None):
        return getSubgeographies_impl(dataset, level)
    else:
        return getSubgeographies_impl(dataset)

def getValue(name):
	return getValue_impl(name)

def returnValue(value):
	returnValue_impl(value);

def ratio(numerator, denominator):
	if denominator == 0: 
		return 0;
	return numerator / float(denominator);

def growth(begin, end, period):
	if begin == 0 or end == 0 or period == 0:
		return 0;

	try:
		rate = pow(end / float(begin), 1 / float(period))
		return (rate - 1) * 100;
	except:
		return 0;

def getValueFromTable(TableID, FieldToRead, IDField, IDValue):
	return getValueFromTable_impl(TableID, FieldToRead, IDField, IDValue);

m_TotalValue = 0.0;

# example call GetSummarizedRates("M02017h_B", "HH")
def getSummarizedRates(RateName, BaseName):
	return getSummarizedRates_impl(RateName, BaseName);

def calculateMedian(numbers, rangeValues):
    return calculateMedianInternal(numbers, rangeValues, None);


def calculateMedianLinear(numbers, rangeValues):
    usePareto = False;
    return calculateMedianInternal(numbers, rangeValues, usePareto);


def calculateMedianPareto(numbers, rangeValues):
    usePareto = True;
    return calculateMedianInternal(numbers, rangeValues, usePareto);


def calculateMedianInternal(numbers, rangeValues, pUsePareto):
    valuesLowerBound = 0 #getLowerBound(numbers);
    valuesUpperBound = len(numbers) - 1 #getUpperBound(numbers);

    nLength = valuesUpperBound - valuesLowerBound + 1;
    val = [0]*nLength;

    for i in range(valuesLowerBound, valuesUpperBound + 1):
        a = numbers[i];
        val[int(i - valuesLowerBound)] = int(a);

    minValues = [0]*nLength;
    maxValues = [0]*nLength;

    for i in range(valuesLowerBound, valuesUpperBound + 1):
        a = rangeValues[i][0];
        minValues[int(i - valuesLowerBound)] = int(a);

        a = rangeValues[i][1];
        maxValues[int(i - valuesLowerBound)] = int(a);

    medianIndex = findMedianIndex(val, nLength);

    # decide about Pareto/Linear
    usePareto = False;

    # check now if all the number higher than medianIndex are zeroes
    bHigherRangeAreZeroes = True;
    tolerance = 1e-8;
    for i in range(medianIndex + 1, nLength):
        if (fabs(val[i]) > tolerance):
            bHigherRangeAreZeroes = False;
            break;

    if (nLength > 23):
        if ((medianIndex < 2) or (medianIndex >= 18)):
            usePareto = True;
        if (bHigherRangeAreZeroes):
            usePareto = False;
    else:
        if (medianIndex != 0):
            usePareto = True;
        if (bHigherRangeAreZeroes):
            usePareto = False;

    if (pUsePareto != None):
        usePareto = usePareto and pUsePareto; # only use Pareto if advised from outside and the data accepts it

    if (medianIndex == nLength - 1):
        median = minValues[nLength - 1] + 1;
    else:
        if (usePareto is False):
            median = medianUsingLinearInterpolation(val, minValues, maxValues, valuesUpperBound - valuesLowerBound + 1, medianIndex) - 0.5;
        else:
            median = medianUsingParetoInterpolation(val, minValues, maxValues, valuesUpperBound - valuesLowerBound + 1, medianIndex) - 0.5;

    return median;


def medianUsingLinearInterpolation(numbers, minRanges, maxRanges, length, medianIndex):

    minValue = minRanges[medianIndex];
    maxValue = maxRanges[medianIndex];

    # Using Linear interpolation
    lowerCount = 0.0;
    median = 0.0;

    for i in range(0, length):
        if (i < medianIndex):
            lowerCount += numbers[i];

    groupCount = numbers[medianIndex];
    t1 = float(m_TotalValue)/2.0;

    if (fabs(groupCount) < 1e-10):
        return 0.5; # it's zero pop area, it will be corrected with +0.5 to be zero

    t2 = (t1 - float(lowerCount))/float(groupCount);
    t3 = maxValue - minValue;
    median = float(minValue) + (t2 * t3) + 0.5;

    # verify medvalue
    if (minValue == int(median)):
        if (numbers[medianIndex] == 0):
            for i in range(medianIndex+1, length):
                if (numbers[i] != 0):
                    dVal = (minRanges[i] - float(minValue))/2.0;
                    median = minValue + dVal + 0.5;
        else:
            if ((medianIndex != 0) and (numbers[medianIndex-1] == 0)):
                for i in range(medianIndex-1, 0, -1):
                    if (numbers[i] != 0):
                        dVal = (float(minValue) - maxRanges[i])/2.0;
                        median = maxRanges[i] + dVal + 0.5;
                        break;

    return median;


def medianUsingParetoInterpolation(numbers, minRanges, maxRanges, length, medianIndex):

    minValue = minRanges[medianIndex];
    maxValue = maxRanges[medianIndex];

    median = 0;
    # Using Pareto Calc
    index1 = findRangeIndex(minValue, minRanges, maxRanges, length);
    N1 = 0;
    for i in range(index1, length):
        N1 += int(numbers[i]);

    if (N1 == 0):
        return 0.5; # it's a zero-pop area

    index2 = findRangeIndex(maxValue, minRanges, maxRanges, length);
    N2 = 0;

    for i in range(index2, length):
        N2 += int(numbers[i]);

    if (N2 == 0):
        return 0.5; # it's a zero-pop area


    #  We have all we need now to calculate the Pareto Median Interpolation
    point5TimesN = 0.5*m_TotalValue;
    log1 = log(float(point5TimesN/N1));
    log2 = log(float(maxValue)/float(minValue));
    log3 = log(float(N2)/float(N1));
    expVal = exp(log1/log3*log2);

    median = expVal*float(minValue) + 0.5;

    # verify medvalue
    if (minValue == int(median)):
        if (numbers[medianIndex] == 0):
            for i in range(medianIndex+1, length):
                if (numbers[i] != 0):
                    dVal = (minRanges[i] - float(minValue))/2.0;
                    median = minValue + dVal + 0.5;

        else:
            if ((medianIndex != 0) and (numbers[medianIndex-1] == 0)):
                for i in range(medianIndex-1, -1):
                    if (numbers[i] != 0):
                        dVal = (float(minValue) - maxRanges[i])/2.0;
                        median = maxRanges[i] + dVal + 0.5;
                        break;

    return median;


def findMedianIndex(numbers, length):
    global m_TotalValue
    m_TotalValue = 0.0;
    for i in range(0, length):
        m_TotalValue += numbers[i];

    median = float(m_TotalValue+1.0)/2.0;
    curCount = 0;
    for i in range(0, length):
        curCount += numbers[i];
        if (curCount >= median):
            return i;

    return 0;


# Determine in what range does this value fall in.
def findRangeIndex(value, minRanges, maxRanges, length):
    #See in what range does this income  fall in?
    for i in range(0, length):
        if ((value >= minRanges[i]) and (value < maxRanges[i])):
            return i;

    return -1;  # Error condition


def getUpperBound(numbers):
    if not numbers:
        return -1;

    upperBound = 0;
    for i in range(1, len(numbers)):
        if numbers[upperBound] < numbers[i]:
            upperBound = i;

    return upperBound;


def getLowerBound(numbers):
    if not numbers:
        return -1;

    lowerBound = 0;
    for i in range(1, len(numbers)):
        if numbers[lowerBound] > numbers[i]:
            lowerBound = i;

    return lowerBound;
