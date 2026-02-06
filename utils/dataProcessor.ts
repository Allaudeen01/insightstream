
import * as Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { Dataset, ColumnStats, ColumnType } from '../types';

export const parseFile = async (file: File): Promise<Dataset> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      const data = e.target?.result;
      if (!data) return reject("No data read");

      if (file.name.endsWith('.csv')) {
        Papa.parse(data as string, {
          header: true,
          dynamicTyping: true,
          complete: (results) => {
            resolve(processRawData(file.name, results.data));
          }
        });
      } else {
        const workbook = XLSX.read(data, { type: 'binary' });
        const sheetName = workbook.SheetNames[0];
        const json = XLSX.utils.sheet_to_json(workbook.Sheets[sheetName]);
        resolve(processRawData(file.name, json));
      }
    };

    if (file.name.endsWith('.csv')) {
      reader.readAsText(file);
    } else {
      reader.readAsBinaryString(file);
    }
  });
};

const processRawData = (name: string, rawData: any[]): Dataset => {
  if (rawData.length === 0) throw new Error("Dataset is empty");
  
  const headers = Object.keys(rawData[0]);
  const columns: ColumnStats[] = headers.map(header => {
    const values = rawData.map(row => row[header]).filter(v => v !== null && v !== undefined && v !== '');
    const type = detectType(values);
    
    const stats: ColumnStats = {
      name: header,
      type,
      missing: rawData.length - values.length,
      unique: new Set(values).size,
    };

    if (type === 'numeric') {
      const numValues = values.map(v => Number(v)).filter(v => !isNaN(v));
      stats.min = Math.min(...numValues);
      stats.max = Math.max(...numValues);
      stats.mean = numValues.reduce((a, b) => a + b, 0) / numValues.length;
      stats.median = numValues.sort((a, b) => a - b)[Math.floor(numValues.length / 2)];
    }

    return stats;
  });

  return {
    name,
    columns,
    data: rawData,
    rowCount: rawData.length
  };
};

const detectType = (values: any[]): ColumnType => {
  if (values.length === 0) return 'string';
  const sample = values.slice(0, 10);
  
  const isDate = sample.every(v => !isNaN(Date.parse(v)) && String(v).length > 5);
  if (isDate) return 'date';

  const isNumeric = sample.every(v => !isNaN(Number(v)));
  if (isNumeric) return 'numeric';

  const uniqueRatio = new Set(values).size / values.length;
  if (uniqueRatio < 0.2 || values.length > 100 && new Set(values).size < 20) return 'categorical';

  return 'string';
};
