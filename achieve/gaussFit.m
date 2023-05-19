function [XC,YC,ZC,FWHMX,FWHMY,FWHMZ] = gaussfit3d(vowel)
    % vowel = vowel/max(vowel(:));
    % Gauss2Mod = @(A,P1,P2,P3,XC,YC,ZC,k,x)...      % P is related to the size of the peak, greater->narrower
    %     A*exp( -( 1./P1.*((x-XC).^2) + 1./P2.*((y-YC).^2) + 1./P3.*((z-ZC).^2)))+k;
    
    GaussMod = @(BETA,x)...      % P is related to the size of the peak, greater->narrower
                BETA(1)*(...
                exp( - 1./BETA(2).*((x(:,1)-BETA(3)).^2 )) .* ...
                exp( - 1./BETA(4).*((x(:,2)-BETA(5)).^2 )) .* ...
                exp( - 1./BETA(6).*((x(:,3)-BETA(7)).^2 ))   ...
                )+BETA(8);
    
    [row,col,depth] = size(vowel);
    axialsum = sum(vowel(round(row/2-1):round(row/2+1),round(col/2-1):round(col/2+1),:),[1,2]);
    [maxv,maxloc] = max(axialsum);
    
    a0 = maxv;
    rx0 = 4;
    ry0 = rx0;
    rz0 = rx0;
    x0 = col/2;
    y0 = row/2;
    z0 = maxloc;
    
    % StartP1 = [a0,(rx0^2)/log(16),x0,min(vowel(:))];
    % StartP2 = [a0,(ry0^2)/log(16),y0,min(vowel(:))];
    % StartP3 = [a0,(rz0^2)/log(16),z0,min(vowel(:))];
    StartP = [a0,(rx0^2)/log(16)*2.35,x0,(ry0^2)/log(16)*2.35,y0,(rz0^2)/log(16)*2.35,z0,min(vowel(:))];
    
    [CoX,CoY,CoZ] = meshgrid(1:1:col,1:1:row,1:1:depth);
    X = [CoX(:),CoY(:),CoZ(:)];
    Y = double(vowel(:));
    
    % figure(333),hold on,
    % cmp = jet(256);
    % scatter(x0,y0,[],cmp(round(z0/depth*256)),'filled')
    % drawnow
    
    try
        beta = nlinfit(X,Y,GaussMod,StartP');
        % disp(beta(2:end)')
        FWHMX = sqrt(beta(2)*log(16))*65;
        FWHMY = sqrt(beta(4)*log(16))*65;
        FWHMZ = sqrt(beta(6)*log(16))*100;
        XC = beta(3);
        YC = beta(5);
        ZC = beta(7);
        % scatter(beta(3),beta(5),[],cmp(1+round(abs(ZC)/depth*256)))
        % hold off
    catch
    
        FWHMX = 0;
        FWHMY = 0;
        FWHMZ = 0;
        XC = -1;
        YC = -1;
        ZC = -1;
        % rethrow(ME)
    end
end
