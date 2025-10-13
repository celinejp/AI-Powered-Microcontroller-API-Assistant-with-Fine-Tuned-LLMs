import React from 'react';
// import { Toaster } from 'react-hot-toast';
// import SimpleCodeGenerator from './components/SimpleCodeGenerator';
import DynamicCodeGenerator from './components/DynamicCodeGenerator';
import './index.css';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#10B981',
              secondary: '#fff',
            },
          },
          error: {
            duration: 5000,
            iconTheme: {
              primary: '#EF4444',
              secondary: '#fff',
            },
          },
        }}
      /> */}
      <DynamicCodeGenerator />
    </div>
  );
}

export default App;
