import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import Papa from 'papaparse';

const FeatureSelector = () => {
  const [columns, setColumns] = useState([]);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [data, setData] = useState(null);
  const [algorithm, setAlgorithm] = useState('kmeans');
  
  useEffect(() => {
    const loadCSV = async () => {
      try {
        const response = await window.fs.readFile('data.csv', { encoding: 'utf8' });
        Papa.parse(response, {
          header: true,
          complete: (results) => {
            setColumns(Object.keys(results.data[0]));
            setData(results.data);
          }
        });
      } catch (error) {
        console.error('Error loading CSV:', error);
      }
    };
    loadCSV();
  }, []);

  const handleFeatureSelect = (column) => {
    if (!selectedFeatures.includes(column)) {
      setSelectedFeatures([...selectedFeatures, column]);
    }
  };

  const handleFeatureRemove = (column) => {
    setSelectedFeatures(selectedFeatures.filter(f => f !== column));
  };

  return (
    <div className="grid grid-cols-3 gap-4">
      <Card className="p-4">
        <h3 className="font-bold mb-4">Available Features</h3>
        <div className="space-y-2">
          {columns.map(column => (
            <div 
              key={column}
              className="p-2 bg-slate-100 rounded cursor-pointer hover:bg-slate-200"
              onClick={() => handleFeatureSelect(column)}
            >
              {column}
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-4">
        <h3 className="font-bold mb-4">Selected Features</h3>
        <div className="space-y-2">
          {selectedFeatures.map(feature => (
            <div 
              key={feature}
              className="p-2 bg-blue-100 rounded cursor-pointer hover:bg-blue-200"
              onClick={() => handleFeatureRemove(feature)}
            >
              {feature}
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-4">
        <h3 className="font-bold mb-4">Algorithm Selection</h3>
        <select 
          className="w-full p-2 border rounded"
          value={algorithm}
          onChange={(e) => setAlgorithm(e.target.value)}
        >
          <option value="kmeans">K-Means Clustering</option>
          <option value="dbscan">DBSCAN</option>
          <option value="hierarchical">Hierarchical Clustering</option>
          <option value="linear">Linear Regression</option>
          <option value="ridge">Ridge Regression</option>
          <option value="lasso">Lasso Regression</option>
        </select>

        <button 
          className="mt-4 w-full p-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          onClick={() => {
            console.log('Selected Features:', selectedFeatures);
            console.log('Algorithm:', algorithm);
            console.log('Data:', data);
          }}
        >
          Run Analysis
        </button>
      </Card>
    </div>
  );
};

export default FeatureSelector;